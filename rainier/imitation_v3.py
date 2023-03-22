import argparse
from collections import defaultdict
from itertools import chain
import json
import logging
import numpy as np
import os
import random
import shutil
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
import transformers
import accelerate
import wandb

from utils.utils import ensure_dir, set_seed, reduce_mean

accelerator = accelerate.Accelerator()
device = accelerator.device
logging.basicConfig(level=logging.INFO)
log = accelerate.logging.get_logger(__name__, log_level='INFO')
def log_info(s):
    if accelerator.is_main_process:
        log.info(s)


class QKADataset(Dataset):
    def __init__(self, args, split, tasks):
        super().__init__()
        self.args = args
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
        for task_ix, task in enumerate(self.tasks):
            path = f'../data/knowledge/knowledge_gkp_gpt3curie.{self.split}.{task}.json'
            with open(path) as f:
                js = json.load(f)
            if self.split == 'train' and self.args.half_half:
                js = js[:len(js) // 2]
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
                        'task_ix': task_ix,
                        'question': q,
                        'choices': choices,
                        'answer': a,
                        'answer_ix': answer_ix,
                        'knowledge': item['knowledges'][k],
                    })
                    # for evaluation, only keep the first knowledge for sake of speed
                    if self.split == 'dev':
                        break
            log_info(f'Loaded dataset for task {task} split {self.split}, skipped {skipped} instances')
        log_info(f'{self.split} set size = {len(ds)}')
        return ds

    @staticmethod
    def collate_fn(batch):
        global tokenizer

        task_ix = torch.tensor([item['task_ix'] for item in batch], dtype=torch.long)
        answer_ix = torch.tensor([item['answer_ix'] for item in batch], dtype=torch.long)

        questions = [item['question'] for item in batch]
        questions_tok = tokenizer.batch_encode_plus(
            questions,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=tokenizer.max_question_len)
        questions_input_ids = questions_tok.input_ids
        questions_attention_mask = questions_tok.attention_mask

        padded_choicess = [item['choices'] + [''] * (8 - len(item['choices'])) for item in batch]
        choicess_tok = [tokenizer.batch_encode_plus(
            padded_choices,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=tokenizer.max_answer_len)
            for padded_choices in padded_choicess]
        choicess_input_ids = torch.stack([choices_tok.input_ids for choices_tok in choicess_tok], dim=0)
        choicess_attention_mask = torch.stack([choices_tok.attention_mask for choices_tok in choicess_tok], dim=0)
        choicess_labels = choicess_input_ids.clone()
        choicess_labels[choicess_attention_mask == 0] = -100

        answers = [item['answer'] for item in batch]
        answers_tok = tokenizer.batch_encode_plus(
            answers,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=tokenizer.max_answer_len)
        answers_input_ids = answers_tok.input_ids
        answers_attention_mask = answers_tok.attention_mask
        answers_labels = answers_input_ids.clone()
        answers_labels[answers_attention_mask == 0] = -100

        knowledges = [item['knowledge'] for item in batch]
        knowledges_with_prefix = [f'Knowledge: {k}' for k in knowledges]
        knowledges_with_prefix_tok = tokenizer.batch_encode_plus(
            knowledges_with_prefix,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=tokenizer.max_knowledge_len)
        knowledges_with_prefix_input_ids = knowledges_with_prefix_tok.input_ids
        knowledges_with_prefix_attention_mask = knowledges_with_prefix_tok.attention_mask
        knowledges_with_prefix_labels = knowledges_with_prefix_input_ids.clone()
        knowledges_with_prefix_labels[knowledges_with_prefix_attention_mask == 0] = -100

        questions_knowledges = [f'{q} \\n {k}' for q, k in zip(questions, knowledges)]
        questions_knowledges_tok = tokenizer.batch_encode_plus(
            questions_knowledges,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=tokenizer.max_question_len + tokenizer.max_knowledge_len)
        questions_knowledges_input_ids = questions_knowledges_tok.input_ids
        questions_knowledges_attention_mask = questions_knowledges_tok.attention_mask

        return {
            'task_ix': task_ix,
            'answer_ix': answer_ix,
            'question_input_ids': questions_input_ids,
            'question_attention_mask': questions_attention_mask,
            'choices_input_ids': choicess_input_ids,
            'choices_attention_mask': choicess_attention_mask,
            'choices_labels': choicess_labels,
            'answer_labels': answers_labels,
            'knowledge_with_prefix_input_ids': knowledges_with_prefix_input_ids,
            'knowledge_with_prefix_attention_mask': knowledges_with_prefix_attention_mask,
            'knowledge_with_prefix_labels': knowledges_with_prefix_labels,
            'question_knowledge_input_ids': questions_knowledges_input_ids,
            'question_knowledge_attention_mask': questions_knowledges_attention_mask,
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
                 eval_accs,
                ):
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.train_sampler = iter(self.train_dataloader)
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer
        if not self.args.nolog and accelerator.is_main_process:
            wandb.init(project='rainier_stageI', name=args.run_name, config=args)
            wandb.define_metric('train/step')
            wandb.define_metric('eval/step')
            wandb.define_metric('train/*', step_metric='train/step')
            wandb.define_metric('eval/*', step_metric='eval/step')

        for _ in range((init_step * args.accumulate_grad_batches) % len(self.train_dataloader)):
            next(self.train_sampler)

        self.eval_losses = eval_losses
        self.eval_accs = eval_accs

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
        self.save(step=step)
        self.valid(step=step)

        accelerator.wait_for_everyone()
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
            qka_loss = torch.tensor(0.0, device=qa_loss.device, dtype=qa_loss.dtype)
            if self.args.qka_loss:
                qka_loss = self.qka_loss(batch)
            loss = qk_loss + qa_loss * self.args.qa_loss_multiplier + qka_loss
            losses.append(loss.detach().clone())
            qk_losses.append(qk_loss.detach().clone())
            qa_losses.append(qa_loss.detach().clone())
            qka_losses.append(qka_loss.detach().clone())
            loss = loss / self.args.accumulate_grad_batches
            accelerator.backward(loss)
        self.optimizer.step()

        loss = torch.stack(losses).mean(dim=0, keepdim=True) # (1)
        qk_loss = torch.stack(qk_losses).mean(dim=0, keepdim=True) # (1)
        qa_loss = torch.stack(qa_losses).mean(dim=0, keepdim=True) # (1)
        qka_loss = torch.stack(qka_losses).mean(dim=0, keepdim=True) # (1)

        losses = accelerator.gather(loss) # (num_gpus)
        qk_losses = accelerator.gather(qk_loss) # (num_gpus)
        qa_losses = accelerator.gather(qa_loss) # (num_gpus)
        qka_losses = accelerator.gather(qka_loss) # (num_gpus)

        loss = losses.mean().item()
        qk_loss = qk_losses.mean().item()
        qa_loss = qa_losses.mean().item()
        qka_loss = qka_losses.mean().item()

        if not self.args.nolog and accelerator.is_main_process:
            if step % self.args.log_interval == 0:
                wandb.log({
                    'train/step': step,
                    'train/loss': loss,
                    'train/qk_loss': qk_loss,
                    'train/qa_loss': qa_loss,
                    'train/qka_loss': qka_loss,
                })

    # The code in this function is adapted from Reward.get_reward() in rainier/rainier/model/reward.py
    def acc(self, batch):
        questions_input_ids = batch['question_input_ids'] # (B, QL)
        questions_attention_mask = batch['question_attention_mask'] # (B, QL)
        choicess_input_ids = batch['choices_input_ids'] # (B, C, AL)
        choicess_attention_mask = batch['choices_attention_mask'] # (B, C, AL)
        choicess_labels = batch['choices_labels'] # (B, C, AL)
        answer_ixs = batch['answer_ix'] # (B)

        # Compute number of choices for each question, and flatten prompts accordingly
        flattened_questions_input_ids = torch.repeat_interleave(questions_input_ids, choicess_input_ids.size(1), dim=0) # (B * C, QL)
        flattened_questions_attention_mask = torch.repeat_interleave(questions_attention_mask, choicess_input_ids.size(1), dim=0) # (B * C, QL)
        flattened_choices_input_ids = choicess_input_ids.flatten(0, 1) # (B * C, AL)
        flattened_choices_attention_mask = choicess_attention_mask.flatten(0, 1) # (B * C, AL)
        flattened_choices_labels = choicess_labels.flatten(0, 1) # (B * C, AL)

        all_losses = []
        assert flattened_choices_input_ids.size(0) % self.args.batch_size == 0        

        for i in range(0, flattened_questions_input_ids.size(0), self.args.batch_size):
            j = min(i + self.args.batch_size, flattened_questions_input_ids.size(0))
            batch_questions_input_ids = flattened_questions_input_ids[i:j]
            batch_questions_attention_mask = flattened_questions_attention_mask[i:j]
            batch_choices_input_ids = flattened_choices_input_ids[i:j]
            batch_choices_attention_mask = flattened_choices_attention_mask[i:j]
            batch_choices_labels = flattened_choices_labels[i:j]

            logits = self.model(
                input_ids=batch_questions_input_ids,
                attention_mask=batch_questions_attention_mask,
                labels=batch_choices_input_ids,
            ).logits # (B, L, V)

            # Loss will be exactly 0 for ignored, pad idx tokens
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            losses = loss_fct(logits.view(-1, logits.size(-1)), batch_choices_labels.view(-1))

            # Take mean of loss
            losses = losses.view(batch_choices_labels.size()) # (B, AL)
            losses = reduce_mean(losses, batch_choices_attention_mask, axis=-1) # (B)

            # Update all loss
            all_losses.append(losses)

        all_losses = torch.cat(all_losses, dim=0) # (B * C)
        all_losses[flattened_choices_labels[:, 0] == 1] = 1e9 # If the first token is [EOS], then the choice is padding
        all_losses = all_losses.view(questions_input_ids.size(0), -1) # (B, C)

        # Now, convert back to tensor of the correct shape - # of questions X max # of answers
        answer_logitss = -all_losses # (B, C)
        answer_probss = answer_logitss.softmax(axis=1)

        # Compute accuracy from argmax answer
        preds = answer_probss.argmax(axis=1)
        corrects = (preds == answer_ixs)

        return {
            'corrects': corrects,
            'preds': preds,
            'answer_logitss': answer_logitss,
            'answer_probss': answer_probss,
        }

    def valid(self, step):
        if self.args.eval_loop_cap is not None and self.args.eval_loop_cap == 0:
            return
        if step % self.args.eval_interval != 0:
            return
        if step in self.eval_losses and step in self.eval_accs:
            return
        log_info(f'Evaluating [step {step}] ...')

        accelerator.wait_for_everyone()
        self.model.eval()

        with torch.no_grad():
            losses, qk_losses, qa_losses, qka_losses = [], [], [], []
            corrects, task_ixs = [], []
            for i, batch in enumerate(tqdm(self.eval_dataloader) if accelerator.is_main_process else self.eval_dataloader):
                if i == self.args.eval_loop_cap:
                    break
                qk_loss = self.qk_loss(batch)
                qa_loss = self.qa_loss(batch)
                qka_loss = torch.tensor(0.0, device=qa_loss.device, dtype=qa_loss.dtype)
                if self.args.qka_loss:
                    qka_loss = self.qka_loss(batch)
                loss = qk_loss + qa_loss * self.args.qa_loss_multiplier + qka_loss
                results = self.acc(batch)
                losses.append(loss.detach().clone())
                qk_losses.append(qk_loss.detach().clone())
                qa_losses.append(qa_loss.detach().clone())
                qka_losses.append(qka_loss.detach().clone())
                corrects.append(results['corrects'])
                task_ixs.append(batch['task_ix'])

        loss = torch.stack(losses).mean(dim=0, keepdim=True) # (1)
        qk_loss = torch.stack(qk_losses).mean(dim=0, keepdim=True) # (1)
        qa_loss = torch.stack(qa_losses).mean(dim=0, keepdim=True) # (1)
        qka_loss = torch.stack(qka_losses).mean(dim=0, keepdim=True) # (1)
        corrects = torch.stack(corrects, dim=0) # (M, B)
        task_ixs = torch.stack(task_ixs, dim=0) # (M, B)

        losses = accelerator.gather(loss) # (num_gpus)
        qk_losses = accelerator.gather(qk_loss) # (num_gpus)
        qa_losses = accelerator.gather(qa_loss) # (num_gpus)
        qka_losses = accelerator.gather(qka_loss) # (num_gpus)
        corrects = accelerator.gather(corrects.unsqueeze(0)) # (num_gpus, M, B)
        task_ixs = accelerator.gather(task_ixs.unsqueeze(0)) # (num_gpus, M, B)

        # Accelerator may pad the tensors to make them divisible by the total batch size
        corrects = corrects.transpose(0, 1).flatten(0, 2)[:len(self.eval_dataloader.dataset)] # (N)
        task_ixs = task_ixs.transpose(0, 1).flatten(0, 2)[:len(self.eval_dataloader.dataset)] # (N)
        
        loss = losses.mean().item()
        qk_loss = qk_losses.mean().item()
        qa_loss = qa_losses.mean().item()
        qka_loss = qka_losses.mean().item()
        acc_weighted = corrects.float().mean().item()
        corrects_by_task = defaultdict(list)
        for task_ix, correct in zip(task_ixs, corrects):
            task = self.eval_dataloader.dataset.tasks[task_ix]
            corrects_by_task[task].append(correct)
        corrects_by_task = {k: torch.stack(v, dim=0) for k, v in corrects_by_task.items()}
        acc_by_task = {k: v.float().mean().item() for k, v in corrects_by_task.items()}
        acc_unweighted = np.mean(list(acc_by_task.values()))

        if not self.args.nolog and accelerator.is_main_process:
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

        # if not self.args.nosave and accelerator.is_main_process:
        #     prev_best_step = None if len(self.eval_losses) == 0 else min(self.eval_losses, key=self.eval_losses.get)
        #     self.eval_losses[step] = loss
        #     if prev_best_step is None or loss < self.eval_losses[prev_best_step]:
        #         if prev_best_step is not None:
        #             try:
        #                 os.remove(f'{self.args.model_dir}/ckp_{prev_best_step}.pth')
        #             except:
        #                 log.warning(f'Cannot remove previous best ckpt!')
        #         shutil.copy(f'{self.args.model_dir}/last.pth', f'{self.args.model_dir}/ckp_{step}.pth')
        #         log_info(f'Best ckpt updated to [step {step}]')
        # else:
        #     self.eval_losses[step] = loss

        if not self.args.nosave and accelerator.is_main_process:
            prev_best_step = None if len(self.eval_accs) == 0 else max(self.eval_accs, key=self.eval_accs.get)
            self.eval_accs[step] = acc_unweighted
            if prev_best_step is None or acc_unweighted > self.eval_accs[prev_best_step]:
                if prev_best_step is not None:
                    try:
                        os.remove(f'{self.args.model_dir}/ckp_{prev_best_step}.pth')
                    except:
                        log.warning(f'Cannot remove previous best ckpt!')
                shutil.copy(f'{self.args.model_dir}/last.pth', f'{self.args.model_dir}/ckp_{step}.pth')
                log_info(f'Best ckpt updated to [step {step}]')
        else:
            self.eval_accs[step] = acc_unweighted

    def save(self, step):
        if self.args.nosave:
            return
        if step % self.args.save_interval != 0:
            return
        # this will overwrite an existing ckpt with the save filename!
        accelerator.wait_for_everyone()
        if accelerator.distributed_type == accelerate.utils.DistributedType.FSDP:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                model_state_dict = self.model.state_dict()
        else:
            model_state_dict = accelerator.unwrap_model(self.model).state_dict() # so that the parameters are synced across GPUs
        accelerator.save({
            'model': model_state_dict,
            'step': step,
            'eval_accs': self.eval_accs,
        }, f'{self.args.model_dir}/last.pth')
        log_info(f'[step {step}] model checkpoint saved')


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
    parser.add_argument('--half_half', default=False, action='store_true')
    parser.add_argument('--qa_loss_multiplier', type=float, default=1.0)

    # other
    parser.add_argument(
        '--log_interval', type=int, default=50, help='step interval to log stats')
    parser.add_argument(
        '--save_interval', type=int, default=1000, help='step interval to save model checkpoints')
    parser.add_argument(
        '--eval_interval', type=int, default=1000, help='step interval to do evaluation')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--nosave', default=False, action='store_true')
    parser.add_argument('--nolog', default=False, action='store_true')
    parser.add_argument('--eval_loop_cap', type=int, default=None, help='cap on number of eval loops')

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    set_seed()

    # Set up save directories
    if not args.nosave:
        args.output_dir = '../runs_stageI/'
        if args.load_from_ckpt is not None:
            args.save_dir = os.path.dirname(os.path.dirname(args.load_from_ckpt))
            args.run_name = args.save_dir.split('/')[-1]
        else:
            args.save_dir = os.path.join(args.output_dir, args.run_name)
        args.model_dir = os.path.join(args.save_dir, 'model')
        if accelerator.is_main_process:
            for d in [args.save_dir, args.model_dir]:
                ensure_dir(d)
    
        log_info(f'Write to output directory: {args.save_dir}')
        if accelerator.is_main_process:
            with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    # Load data
    log_info(f'Loading data ...')
    train_dataset = QKADataset(args, 'train', args.train_tasks)
    eval_dataset = QKADataset(args, 'dev', args.train_tasks)
    # train ds is shuffled in its constructor
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=QKADataset.collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=QKADataset.collate_fn)
    train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)

    # Initialize models and optimizer
    log_info(f'Initializing models ...')
    global tokenizer
    tokenizer = transformers.T5Tokenizer.from_pretrained(args.model_type)
    tokenizer.max_question_len = args.max_question_len
    tokenizer.max_answer_len = args.max_answer_len
    tokenizer.max_knowledge_len = args.max_knowledge_len
    model = transformers.T5ForConditionalGeneration.from_pretrained(args.model_type)
    model = accelerator.prepare(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = accelerator.prepare(optimizer)
    init_step = 0
    eval_losses = {}
    eval_accs = {}

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
        eval_accs=eval_accs,
    )

    # Train
    steps = list(range(init_step, args.total_steps + 1))
    steps = tqdm(steps) if accelerator.is_main_process else steps
    for step in steps:
        trainer.train(step)


if __name__ == '__main__':
    main()
