import argparse
from collections import defaultdict
from datetime import datetime
from itertools import chain
import json
import logging
import numpy as np
import os
import random
from tqdm import tqdm

import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
import transformers
from accelerate import Accelerator
import wandb

from utils.utils import ensure_dir, set_seed, reduce_mean
from data import QADataset

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO'))
log = logging.getLogger(__name__)


class QKADataset(Dataset):
    def __init__(self, split, tasks):
        super().__init__()
        self.split = split
        self.tasks = tasks.split(',')

        self.instances = self.load_datasets()

        if split == 'train':
            random.shuffle(self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

    def load_datasets(self):
        def parse_choices(s):
            '''
            s: serialized_choices '(A) ... (B) ... (C) ...'
            '''
            choices = []
            key = 'A' if s.find('(A)') != -1 else 'a'
            while True:
                pos = s.find(f'({chr(ord(key) + 1)})')
                if pos == -1:
                    break
                choice = s[3:pos]
                s = s[pos:]
                choice = choice.strip(' ')
                choices.append(choice)
                key = chr(ord(key) + 1)
            choice = s[3:]
            choice = choice.strip(' ')
            choices.append(choice)
            return choices

        ds = []
        for task in self.tasks:
            path = f'../data/knowledge/knowledge_gkp_gpt3curie.{self.split}.{task}.json'
            with open(path) as f:
                js = json.load(f)
            skipped = 0
            for item in js:
                try:
                    q, a = item['query'], item['answer']
                    choices = parse_choices(q.split('\\n')[1].strip(' '))
                    answer_ix = choices.index(a)
                except Exception as e:
                    skipped += 1
                    continue
                for k in range(len(item['knowledges'])):
                    ds.append({
                        'task': task,
                        'question': q,
                        'choices': choices,
                        'answer': a,
                        'answer_ix': answer_ix,
                        'knowledge': item['knowledges'][k],
                    })
                    # for evaluation, only keep the first knowledge for sake of speed
                    if self.split == 'dev':
                        break
            print(f'Loaded dataset for task {task} split {self.split}, skipped {skipped} instances')
        print(f'{self.split} set size = {len(ds)}')
        return ds

    @staticmethod
    def collate_fn(batch):
        global tokenizer

        task = [item['task'] for item in batch]
        answer_ix = torch.tensor([item['answer_ix'] for item in batch], dtype=torch.long)

        question = [item['question'] for item in batch]
        question_tok = tokenizer.batch_encode_plus(
            question,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=tokenizer.max_question_len)

        choices = [item['choices'] for item in batch]
        choices_tok = [tokenizer.batch_encode_plus(
            item['choices'],
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=tokenizer.max_answer_len)
            for item in batch]
        choices_labels = [choice_tok.input_ids.clone() for choice_tok in choices_tok]
        for choice_labels, choice_tok in zip(choices_labels, choices_tok):
            choice_labels[choice_tok.attention_mask == 0] = -100

        answer = [item['answer'] for item in batch]
        answer_tok = tokenizer.batch_encode_plus(
            answer,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=tokenizer.max_answer_len)
        answer_labels = answer_tok.input_ids.clone()
        answer_labels[answer_tok.attention_mask == 0] = -100

        knowledge = [item['knowledge'] for item in batch]
        knowledge_with_prefix = [f'Knowledge: {k}' for k in knowledge]
        knowledge_with_prefix_tok = tokenizer.batch_encode_plus(
            knowledge_with_prefix,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=tokenizer.max_knowledge_len)
        knowledge_with_prefix_labels = knowledge_with_prefix_tok.input_ids.clone()
        knowledge_with_prefix_labels[knowledge_with_prefix_tok.attention_mask == 0] = -100

        question_knowledge = [f'{q} \\n {k}' for q, k in zip(question, knowledge)]
        question_knowledge_tok = tokenizer.batch_encode_plus(
            question_knowledge,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=tokenizer.max_question_len + tokenizer.max_knowledge_len)

        return {
            'task': task,
            'answer_ix': answer_ix,
            'question_input_ids': question_tok.input_ids,
            'question_attention_mask': question_tok.attention_mask,
            'choices_input_ids': [choice_tok.input_ids for choice_tok in choices_tok],
            'choices_attention_mask': [choice_tok.attention_mask for choice_tok in choices_tok],
            'choices_labels': choices_labels,
            'answer_input_ids': answer_tok.input_ids,
            'answer_attention_mask': answer_tok.attention_mask,
            'answer_labels': answer_labels,
            'knowledge_with_prefix_input_ids': knowledge_with_prefix_tok.input_ids,
            'knowledge_with_prefix_attention_mask': knowledge_with_prefix_tok.attention_mask,
            'knowledge_with_prefix_labels': knowledge_with_prefix_labels,
            'question_knowledge_input_ids': question_knowledge_tok.input_ids,
            'question_knowledge_attention_mask': question_knowledge_tok.attention_mask,
        }

class Trainer:
    def __init__(self,
                 args,
                 train_dataloader,
                 eval_dataloader,
                 tokenizer,
                 model,
                 optimizer,
                 init_step,
                 eval_losses,
                 device,
                 accelerator,
                ):
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.train_sampler = iter(self.train_dataloader)
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.accelerator = accelerator
        if not args.nosave and accelerator.is_main_process:
            # self.writer = SummaryWriter(log_dir=args.tensorboard_dir)
            wandb.init(project='rainier_stageI', name=args.run_name, config=args)
            wandb.define_metric('train/step')
            wandb.define_metric('eval/step')
            wandb.define_metric('train/*', step_metric='train/step')
            wandb.define_metric('eval/*', step_metric='eval/step')

        for _ in range((init_step * args.accumulate_grad_batches) % len(self.train_dataloader)):
            next(self.train_sampler)

        self.eval_losses = eval_losses

    def qk_loss(self, batch):
        logits = self.model(
            input_ids=batch['question_input_ids'],
            attention_mask=batch['question_attention_mask'],
            labels=batch['knowledge_with_prefix_labels'],
        ).logits # (B, L, V)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        losses = loss_fn(logits.view(-1, logits.size(-1)), batch['knowledge_with_prefix_labels'].view(-1))
        losses = losses.view(batch['knowledge_with_prefix_labels'].shape) # (B, L)
        loss_mask = batch['knowledge_with_prefix_attention_mask'].clone()
        loss_mask[:, :2] = 0 # Because "Knowledege:" is two tokens
        losses = reduce_mean(losses, loss_mask, axis=-1) # (B)
        loss = losses.mean()

        return loss

    def qa_loss(self, batch):
        loss = self.model(
            input_ids=batch['question_input_ids'],
            attention_mask=batch['question_attention_mask'],
            labels=batch['answer_labels'],
        ).loss

        return loss

    def qka_loss(self, batch):
        loss = self.model(
            input_ids=batch['question_knowledge_input_ids'],
            attention_mask=batch['question_knowledge_attention_mask'],
            labels=batch['answer_labels'],
        ).loss

        return loss

    def train(self, step):
        self.eval(step=step)
        self.save(step=step)

        self.model.train()
        self.optimizer.zero_grad()
        losses, qk_losses, qa_losses, qka_losses = [], [], [], []
        for _ in range(self.args.accumulate_grad_batches):
            try:
                batch = next(self.train_sampler)
            except StopIteration:
                self.train_sampler = iter(self.train_dataloader)
                batch = next(self.train_sampler)
            qk_loss = self.qk_loss(batch)
            qa_loss = self.qa_loss(batch)
            qka_loss = torch.tensor(0.0, device=qk_loss.device)
            if self.args.qka_loss:
                qka_loss = self.qka_loss(batch)
            loss = qk_loss + qa_loss + qka_loss
            losses.append(loss.item())
            qk_losses.append(qk_loss.item())
            qa_losses.append(qa_loss.item())
            qka_losses.append(qka_loss.item())
            loss /= self.args.accumulate_grad_batches
            self.accelerator.backward(loss)
        self.optimizer.step()

        loss = np.mean(losses)
        qk_loss = np.mean(qk_losses)
        qa_loss = np.mean(qa_losses)
        qka_loss = np.mean(qka_losses)

        if not self.args.nosave and self.accelerator.is_main_process:
            if step % self.args.log_interval == 0:
                # self.writer.add_scalar('train/loss', loss.item(), step)
                wandb.log({
                    'train/step': step,
                    'train/loss': loss,
                    'train/qk_loss': qk_loss,
                    'train/qa_loss': qa_loss,
                    'train/qka_loss': qka_loss,
                })

    # The code in this function is adapted from Reward.get_reward() in rainier/rainier/model/reward.py
    def acc(self, batch):
        questions_input_ids = batch['question_input_ids'] # (B, L)
        questions_attention_mask = batch['question_attention_mask'] # (B, L)
        choicess_input_ids = batch['choices_input_ids'] # [(K, L)]
        choicess_attention_mask = batch['choices_attention_mask'] # [(K, L)]
        choicess_labels = batch['choices_labels'] # [(K, L)]
        answer_ixs = batch['answer_ix'] # (B)

        # Compute number of choices for each question, and flatten prompts accordingly
        num_ans = torch.tensor([choices_input_ids.size(0) for choices_input_ids in choicess_input_ids], device=choicess_input_ids[0].device)
        max_ans_num = num_ans.max().item()
        flattened_questions_input_ids = torch.repeat_interleave(questions_input_ids, num_ans, dim=0) # (B * K, L)
        flattened_questions_attention_mask = torch.repeat_interleave(questions_attention_mask, num_ans, dim=0) # (B * K, L)
        flattened_choices_input_ids = torch.cat(choicess_input_ids, dim=0) # (B * K, L)
        flattened_choices_attention_mask = torch.cat(choicess_attention_mask, dim=0) # (B * K, L)
        flattened_choices_labels = torch.cat(choicess_labels, dim=0) # (B * K, L)

        # Preallocate tensor for all of the loss
        all_losses = torch.zeros(flattened_questions_input_ids.size(0), device=self.device)

        for i in range(0, flattened_questions_input_ids.size(0), self.args.batch_size):
            j = min(i + self.args.batch_size, flattened_questions_input_ids.size(0))
            batch_questions_input_ids = flattened_questions_input_ids[i:j]
            batch_questions_attention_mask = flattened_questions_attention_mask[i:j]
            batch_choices_input_ids = flattened_choices_input_ids[i:j]
            batch_choices_attention_mask = flattened_choices_attention_mask[i:j]
            batch_choices_labels = flattened_choices_labels[i:j]

            # Tokenize prompts and inputs
            # tokenized_prompts = self.tokenizer(batch_prompts, return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.args.max_question_len).to(self.device)
            # tokenized_choices = self.tokenizer(batch_choices, return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.args.max_answer_len).to(self.device)
            # tokenized_choices_ids = tokenized_choices.input_ids
            # pad_mask = (tokenized_choices_ids == self.tokenizer.pad_token_id)

            with torch.no_grad():
                logits = self.model(
                    input_ids=batch_questions_input_ids,
                    attention_mask=batch_questions_attention_mask,
                    labels=batch_choices_input_ids,
                ).logits # (B, L, V)

            # Set ignore index for loss calculation
            # tokenized_choices_ids[pad_mask] = -100

            # Loss will be exactly 0 for ignored, pad idx tokens
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100,reduction='none')
            losses = loss_fct(logits.view(-1, logits.size(-1)), batch_choices_labels.view(-1))

            # Take mean of loss
            losses = losses.view(batch_choices_labels.shape) # (B, L)
            losses = reduce_mean(losses, batch_choices_attention_mask, axis=-1) # (B)

            # Update all loss
            all_losses[i:j] = losses

        # Now, convert back to tensor of the correct shape - # of questions X max # of answers
        answer_logits = torch.empty(questions_input_ids.size(0), max_ans_num, device=self.device).fill_(float('-inf'))
        cur_arr_idx = 0
        for idx, sz in enumerate(num_ans):
            answer_logits[idx, :sz] = -all_losses[cur_arr_idx:cur_arr_idx+sz]
            cur_arr_idx += sz
        answer_probs = answer_logits.softmax(axis=1)

        # Compute accuracy from argmax answer
        preds = answer_probs.argmax(axis=1)
        corrects = (preds == answer_ixs).tolist()

        return {
            'corrects': corrects,
            'preds': preds,
            'answer_logits': answer_logits.detach().cpu(),
            'answer_probs': answer_probs.detach().cpu(),
        }

    def eval(self, step):
        if step == 0 and self.args.skip_init_eval:
            return
        if step % self.args.eval_interval != 0:
            return
        if step in self.eval_losses:
            return
        log.info(f'Evaluating [step {step}] ...')

        losses = []
        qk_losses = []
        qa_losses = []
        qka_losses = []
        corrects = []
        corrects_by_task = defaultdict(list)
        for i, batch in enumerate(tqdm(self.eval_dataloader)):
            self.model.eval()
            with torch.no_grad():
                qk_loss = self.qk_loss(batch)
                qa_loss = self.qa_loss(batch)
                qka_loss = self.qka_loss(batch)
                loss = qk_loss + qa_loss + qka_loss
                results = self.acc(batch)
            losses.append(loss.item())
            qk_losses.append(qk_loss.item())
            qa_losses.append(qa_loss.item())
            qka_losses.append(qka_loss.item())
            corrects += results['corrects']
            for task, c in zip(batch['task'], results['corrects']):
                corrects_by_task[task].append(c)

        loss = np.mean(losses)
        qk_loss = np.mean(qk_losses)
        qa_loss = np.mean(qa_losses)
        qka_loss = np.mean(qka_losses)
        acc_weighted = np.mean(corrects)
        acc_by_task = {k: np.mean(v) for k, v in corrects_by_task.items()}
        acc_unweighted = np.mean(list(acc_by_task.values()))

        if not self.args.nosave and self.accelerator.is_main_process:
            # self.writer.add_scalar('eval/loss', loss, step)
            stats = {
                'eval/step': step,
                'eval/loss': loss,
                'eval/qk_loss': qk_loss,
                'eval/qa_loss': qa_loss,
                'eval/qka_loss': qka_loss,
                'eval/acc_weighted': acc_weighted,
                'eval/acc_unweighted': acc_unweighted,
            }
            for task, acc in acc_by_task.items():
                stats[f'eval/acc/{task}'] = acc
            wandb.log(stats)

            prev_best_step = None if len(self.eval_losses) == 0 else min(self.eval_losses, key=self.eval_losses.get)
            self.eval_losses[step] = qk_loss
            if prev_best_step is None or qk_loss < self.eval_losses[prev_best_step]:
                if prev_best_step is not None:
                    try:
                        os.remove(f'{self.args.model_dir}/ckp_{prev_best_step}.pth')
                    except:
                        log.warning(f'Cannot remove previous best ckpt!')
                self.save(step, last=False)
                log.info(f'Best ckpt updated to [step {step}]')
        else:
            self.eval_losses[step] = qk_loss

    def save(self, step, last=True):
        if self.args.nosave:
            return
        if step % self.args.save_interval != 0:
            return
        # this will overwrite an existing ckpt with the save filename!
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': step,
            'eval_losses': self.eval_losses,
        }, f'{self.args.model_dir}/{"last" if last else "ckp_" + str(step)}.pth')
        log.info(f'[step {step}] model checkpoint saved')

def get_args():
    parser = argparse.ArgumentParser()

    # common
    parser.add_argument('--model_type', type=str, default='t5-large', choices=['t5-large', 'allenai/unifiedqa-t5-large', 'allenai/unifiedqa-t5-11b', 'allenai/unifiedqa-v2-t5-large-1251000'])
    parser.add_argument('--load_from_ckpt', default=None)
    parser.add_argument('--max_question_len', type=int, default=256)
    parser.add_argument('--max_knowledge_len', type=int, default=32)
    parser.add_argument('--max_answer_len', type=int, default=128)

    # train
    parser.add_argument('--data_path', type=str, default='../data/{datapath}/{split}.tsv')
    parser.add_argument('--train_tasks', type=str, default='obqa,arc_e,arc_h,ai2sci_e,ai2sci_m,csqa,qasc,piqa,siqa,wg')
    parser.add_argument('--total_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--qka_loss', default=False, action='store_true')

    # other
    parser.add_argument(
        '--log_interval', type=int, default=50, help='step interval to log stats')
    parser.add_argument(
        '--save_interval', type=int, default=1000, help='step interval to save model checkpoints')
    parser.add_argument(
        '--eval_interval', type=int, default=1000, help='step interval to do evaluation')
    parser.add_argument('--nosave', default=False, action='store_true')
    parser.add_argument('--job_name', type=str, default=None)
    parser.add_argument('--skip_init_eval', default=False, action='store_true')

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    set_seed()

    # GPUs
    accelerator = Accelerator()
    device = accelerator.device
    '''
    assert torch.cuda.is_available(), 'CUDA is not available'
    num_gpus = torch.cuda.device_count()
    log.info(f'Detected {num_gpus} GPUS')
    devices = {}
    for i in range(num_gpus):
        devices[i] = torch.device('cuda:' + str(i))

    device_map = None
    if num_gpus == 1:
        device_map = None
    elif num_gpus == 4:  # 4x RTX6000 for T5-large
        device_map = {
            0: [0, 1, 2],
            1: [3, 4, 5, 6, 7, 8, 9],
            2: [10, 11, 12, 13, 14, 15, 16],
            3: [17, 18, 19, 20, 21, 22, 23],
        }
    elif num_gpus == 6:
        device_map = {
            0: [0],
            1: [1, 2, 3, 4],
            2: [5, 6, 7, 8],
            3: [9, 10, 11, 12, 13],
            4: [14, 15, 16, 17, 18],
            5: [19, 20, 21, 22, 23],
        }
    elif num_gpus == 8:  # 8x V100 for T5-3b
        device_map = {
            0: [0],
            1: [1, 2, 3],
            2: [4, 5, 6],
            3: [7, 8, 9],
            4: [10, 11, 12],
            5: [13, 14, 15],
            6: [16, 17, 18, 19],
            7: [20, 21, 22, 23],
        }
    else:
        log.error('Invalid number of GPUs!')
        exit(-1)
    '''

    # Set up save directories
    if not args.nosave:
        args.output_dir = '../runs_stageI/'
        if args.load_from_ckpt is not None:
            args.save_dir = os.path.dirname(os.path.dirname(args.load_from_ckpt))
            args.run_name = args.save_dir.split('/')[-1]
        else:
            time = datetime.now()
            date_time = time.strftime('%Y%m%d-%H%M%S')
            import socket
            args.run_name = date_time + '_' + socket.gethostname() + '_' + args.job_name
            args.save_dir = os.path.join(args.output_dir, args.run_name)
        args.model_dir = os.path.join(args.save_dir, 'model')
        args.tensorboard_dir = os.path.join(args.save_dir, 'tensorboard')
        for d in [args.save_dir, args.model_dir, args.tensorboard_dir]:
            ensure_dir(d)
    
        log.info(f'Write to output directory: {args.save_dir}')
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    # Load data
    log.info(f'Loading data ...')
    train_dataset = QKADataset('train', args.train_tasks)
    eval_dataset = QKADataset('dev', args.train_tasks)
    # train ds is shuffled in its constructor
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=QKADataset.collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=QKADataset.collate_fn)

    # Initialize models and optimizer
    log.info(f'Initializing models ...')
    global tokenizer
    tokenizer = transformers.T5Tokenizer.from_pretrained(args.model_type)
    tokenizer.max_question_len = args.max_question_len
    tokenizer.max_answer_len = args.max_answer_len
    tokenizer.max_knowledge_len = args.max_knowledge_len
    model = transformers.T5ForConditionalGeneration.from_pretrained(args.model_type)
    model.to(device)
    # model.parallelize(device_map=device_map)
    model = accelerator.prepare(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    init_step = 0
    eval_losses = {}

    # Load from checkpoint if continue training
    if args.load_from_ckpt is not None:
        checkpoint = torch.load(args.load_from_ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        init_step = checkpoint['step']
        eval_losses = checkpoint['eval_losses']
        checkpoint.clear()

    optimizer, train_dataloader, eval_dataloader = accelerator.prepare(optimizer, train_dataloader, eval_dataloader)

    # Set up trainer
    trainer = Trainer(
        args=args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        model=model,
        optimizer=optimizer,
        init_step=init_step,
        eval_losses=eval_losses,
        device=device,
        accelerator=accelerator,
    )

    # Train
    pbar = tqdm(list(range(init_step, args.total_steps + 1)))
    for step in pbar:
        trainer.train(step)


if __name__ == '__main__':
    main()

