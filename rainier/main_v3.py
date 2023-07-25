import argparse
from collections import defaultdict
import copy
from itertools import chain
import json
import logging
import numpy as np
import os
import random
import shutil
from tqdm import tqdm
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
import transformers
import accelerate
import wandb

from utils.utils import ensure_dir, set_seed, reduce_mean, reduce_sum, ceil_div, whiten, clamp
from args import get_args
from model.policy import Policy
from model.value import Value
from model.reward import Reward
from ppo import PPOTrainer

accelerator = accelerate.Accelerator()
device = accelerator.device
logging.basicConfig(level=logging.INFO)
log = accelerate.logging.get_logger(__name__, log_level='INFO')
def log_info(s):
    if accelerator.is_main_process:
        log.info(s)


datapath_by_task_and_split = {
    'obqa': {'default': 'uqa/openbookqa'},
    'arc_e': {'default': 'uqa/arc_easy'},
    'arc_h': {'default': 'uqa/arc_hard'},
    'ai2sci_e': {'default': 'uqa/ai2_science_elementary'},
    'ai2sci_m': {'default': 'uqa/ai2_science_middle'},
    'csqa': {'default': 'uqa/commonsenseqa'},
    'qasc': {'default': 'uqa/qasc'},
    'piqa': {'default': 'uqa/physical_iqa', 'test': 'uqa/physical_iqa_test'},
    'siqa': {'default': 'uqa/social_iqa', 'test': 'uqa/social_iqa_test'},
    'wg': {'default': 'uqa/winogrande_xl'},

    'numersense': {'default': 'numersense'},
    'riddlesense': {'default': 'riddlesense'},
    'quartz': {'default': 'quartz'},
    'hellaswag': {'default': 'hellaswag'},

    'sciq': {'default': 'sciq'},
    'quarel': {'default': 'quarel'},
    'quartz': {'default': 'quartz'},
    'wsc273': {'default': 'wsc273'},
    'copa': {'default': 'copa'},
    'numersense_new': {'default': 'numersense_new'},
    # 'truthfulqa_mc1': {'default': 'truthfulqa_mc1'},
}

# This does not lowercase the data, by default
class QADataset(Dataset):
    def __init__(self, args, split, tasks, data_path, tokenizer, subsplit=None):
        super().__init__()
        self.args = args
        self.split = split
        self.tasks = tasks.split(',')
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.subsplit = subsplit

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

        instances = []
        if self.data_path.endswith('.tsv'):
            for task_ix, task in enumerate(self.tasks):
                skipped = 0
                datapath_by_split = datapath_by_task_and_split[task]
                datapath = datapath_by_split[self.split if self.split in datapath_by_split else 'default']
                with open(self.data_path.replace('{datapath}', datapath).replace('{split}', self.split)) as f:
                    lines = f.readlines()
                    if self.subsplit is not None:
                        if self.subsplit == 'qa':
                            lines = lines[:len(lines) // 2]
                        elif self.subsplit == 'rl':
                            lines = lines[len(lines) // 2:]
                        else:
                            raise
                    for line in lines:
                        try:
                            q, a = line.strip('\n').split('\t')
                            q = q.strip(' ')
                            a = a.strip(' ')
                            choices = parse_choices(q.split('\\n')[1].strip(' '))
                            answer_ix = choices.index(a)
                        except Exception as e:
                            skipped += 1
                            continue
                        instances.append({
                            'task': task,
                            'task_ix': task_ix,
                            'split': self.split,
                            'question': q,
                            'choices': choices,
                            'answer': a,
                            'answer_ix': answer_ix,
                        })
                log_info(f'Loaded dataset for task {task} split {self.split}, skipped {skipped} instances')
        elif self.data_path.endswith('.json') and ('{task}' in self.data_path and '{split}' in self.data_path):
            for task_ix, task in enumerate(self.tasks):
                skipped = 0
                with open(self.data_path.replace('{task}', task).replace('{split}', self.split)) as f:
                    js = json.load(f)
                    if self.subsplit is not None:
                        if self.subsplit == 'qa':
                            js = js[:len(js) // 2]
                        elif self.subsplit == 'rl':
                            js = js[len(js) // 2:]
                        else:
                            raise
                    for item in js:
                        try:
                            q, a = item['query'], item['answer']
                            choices = parse_choices(q.split('\\n')[1].strip(' '))
                            answer_ix = choices.index(a)
                            knowledges = item['knowledges'][:self.args.num_samples]
                        except Exception as e:
                            skipped += 1
                            continue
                        instances.append({
                            'task': task,
                            'task_ix': task_ix,
                            'split': self.split,
                            'question': q,
                            'choices': choices,
                            'answer': a,
                            'answer_ix': answer_ix,
                            'knowledges': knowledges,
                        })
                log_info(f'Loaded dataset for task {task} split {self.split}, skipped {skipped} instances')
        elif self.data_path.endswith('.json'):
            with open(self.data_path) as f:
                js = json.load(f)
                for item in js:
                    q, a = item['query'], item['answer']
                    choices = item['cands']
                    answer_ix = item['answer_ix']
                    knowledges = item['knowledges'][:self.args.num_samples]
                    instances.append({
                        'task': item['task'],
                        'task_ix': self.tasks.index(item['task']),
                        'split': self.split,
                        'question': q,
                        'choices': choices,
                        'answer': a,
                        'answer_ix': answer_ix,
                        'knowledges': knowledges,
                    })
        log_info(f'Loaded split {self.split} with {len(instances)} total instances')
        return instances

    # Make a collate function to fix dataloader weird list batching
    def collate_fn(self, batch):
        task_ixs = torch.tensor([item['task_ix'] for item in batch], dtype=torch.long)
        answer_ixs = torch.tensor([item['answer_ix'] for item in batch], dtype=torch.long)

        questions = [item['question'] for item in batch]
        questions_tok = self.tokenizer.batch_encode_plus(
            questions,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.tokenizer.max_question_len)
        questions_input_ids = questions_tok.input_ids
        questions_attention_mask = questions_tok.attention_mask

        CAP = 8 if batch[0]['split'] == 'train' else 16
        choicess = [item['choices'] + [''] * (CAP - len(item['choices'])) for item in batch]
        choicess_tok = [self.tokenizer.batch_encode_plus(
            choices,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.tokenizer.max_answer_len)
            for choices in choicess]
        choicess_input_ids = torch.stack([choices_tok.input_ids for choices_tok in choicess_tok], dim=0)
        choicess_attention_mask = torch.stack([choices_tok.attention_mask for choices_tok in choicess_tok], dim=0)
        choicess_labels = choicess_input_ids.clone()
        choicess_labels[choicess_attention_mask == 0] = -100

        answers = [item['answer'] for item in batch]
        answers_tok = self.tokenizer.batch_encode_plus(
            answers,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.tokenizer.max_answer_len)
        answers_input_ids = answers_tok.input_ids
        answers_attention_mask = answers_tok.attention_mask
        answers_labels = answers_input_ids.clone()
        answers_labels[answers_attention_mask == 0] = -100

        lowercased_questions = [question.lower() for question in questions]
        lowercased_questions_tok = self.tokenizer.batch_encode_plus(
            lowercased_questions,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.tokenizer.max_question_len)
        lowercased_questions_input_ids = lowercased_questions_tok.input_ids
        lowercased_questions_attention_mask = lowercased_questions_tok.attention_mask

        lowercased_choicess = [[choice.lower() for choice in choices] for choices in choicess]
        lowercased_choicess_tok = [self.tokenizer.batch_encode_plus(
            choices,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.tokenizer.max_answer_len)
            for choices in lowercased_choicess]
        lowercased_choicess_input_ids = torch.stack([choices_tok.input_ids for choices_tok in lowercased_choicess_tok], dim=0)
        lowercased_choicess_attention_mask = torch.stack([choices_tok.attention_mask for choices_tok in lowercased_choicess_tok], dim=0)
        lowercased_choicess_labels = lowercased_choicess_input_ids.clone()
        lowercased_choicess_labels[lowercased_choicess_attention_mask == 0] = -100

        lowercased_answers = [answer.lower() for answer in answers]
        lowercased_answers_tok = self.tokenizer.batch_encode_plus(
            lowercased_answers,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.tokenizer.max_answer_len)
        lowercased_answers_input_ids = lowercased_answers_tok.input_ids
        lowercased_answers_attention_mask = lowercased_answers_tok.attention_mask
        lowercased_answers_labels = lowercased_answers_input_ids.clone()
        lowercased_answers_labels[lowercased_answers_attention_mask == 0] = -100

        result = {
            'task_ixs': task_ixs,
            'answer_ixs': answer_ixs,
            'questions_text': questions,
            'questions_input_ids': questions_input_ids,
            'questions_attention_mask': questions_attention_mask,
            'choicess_text': choicess,
            'choicess_input_ids': choicess_input_ids,
            'choicess_attention_mask': choicess_attention_mask,
            'choicess_labels': choicess_labels,
            'answers_labels': answers_labels,
            'lowercased_questions_text': lowercased_questions,
            'lowercased_questions_input_ids': lowercased_questions_input_ids,
            'lowercased_questions_attention_mask': lowercased_questions_attention_mask,
            'lowercased_choicess_input_ids': lowercased_choicess_input_ids,
            'lowercased_choicess_attention_mask': lowercased_choicess_attention_mask,
            'lowercased_choicess_labels': lowercased_choicess_labels,
            'lowercased_answers_labels': lowercased_answers_labels,
        }

        if 'knowledges' in batch[0]:
            knowledgess = [item['knowledges'] + [''] * (self.args.num_samples - len(item['knowledges'])) for item in batch]
            knowledgess = list(map(list, zip(*knowledgess))) # transpose the list
            knowledgess_tok = [self.tokenizer.batch_encode_plus(
                knowledges,
                return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.tokenizer.max_knowledge_len)
                for knowledges in knowledgess]
            knowledgess_input_ids = torch.stack([knowledges_tok.input_ids for knowledges_tok in knowledgess_tok], dim=0)
            knowledgess_attention_mask = torch.stack([knowledges_tok.attention_mask for knowledges_tok in knowledgess_tok], dim=0)
            knowledgess_labels = knowledgess_input_ids.clone()
            knowledgess_labels[knowledgess_attention_mask == 0] = -100

            lowercased_knowledgess = [[knowledge.lower() for knowledge in knowledges] for knowledges in knowledgess]
            lowercased_knowledgess_tok = [self.tokenizer.batch_encode_plus(
                knowledges,
                return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.tokenizer.max_knowledge_len)
                for knowledges in lowercased_knowledgess]
            lowercased_knowledgess_input_ids = torch.stack([knowledges_tok.input_ids for knowledges_tok in lowercased_knowledgess_tok], dim=0)
            lowercased_knowledgess_attention_mask = torch.stack([knowledges_tok.attention_mask for knowledges_tok in lowercased_knowledgess_tok], dim=0)
            lowercased_knowledgess_labels = lowercased_knowledgess_input_ids.clone()
            lowercased_knowledgess_labels[lowercased_knowledgess_attention_mask == 0] = -100

            prefixed_knowledgess = [[f'Knowledge: {k}' for k in ks] for ks in knowledgess]
            prefixed_knowledgess_tok = [self.tokenizer.batch_encode_plus(
                prefixed_knowledges,
                return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.tokenizer.max_knowledge_len)
                for prefixed_knowledges in prefixed_knowledgess]
            prefixed_knowledgess_input_ids = torch.stack([prefixed_knowledges_tok.input_ids for prefixed_knowledges_tok in prefixed_knowledgess_tok], dim=0)
            prefixed_knowledgess_attention_mask = torch.stack([prefixed_knowledges_tok.attention_mask for prefixed_knowledges_tok in prefixed_knowledgess_tok], dim=0)
            prefixed_knowledgess_labels = prefixed_knowledgess_input_ids.clone()
            prefixed_knowledgess_labels[prefixed_knowledgess_attention_mask == 0] = -100

            result.update({
                'knowledgess_text': knowledgess,
                'knowledgess_input_ids': knowledgess_input_ids, # (K, B, KL)
                'knowledegss_attention_mask': knowledgess_attention_mask,
                'knowledgess_labels': knowledgess_labels,
                'lowercased_knowledgess_text': lowercased_knowledgess,
                'lowercased_knowledgess_input_ids': lowercased_knowledgess_input_ids,
                'lowercased_knowledgess_attention_mask': lowercased_knowledgess_attention_mask,
                'lowercased_knowledgess_labels': lowercased_knowledgess_labels,
                'prefixed_knowledgess_input_ids': prefixed_knowledgess_input_ids,
                'prefixed_knowledgess_attention_mask': prefixed_knowledgess_attention_mask,
                'prefixed_knowledgess_labels': prefixed_knowledgess_labels,
            })
        
        return result


class PPOTrainer:
    def __init__(self,
                 args: argparse.Namespace,
                 train_dataloader: Optional[DataLoader],
                 train_dataloader_qa: Optional[DataLoader],
                 train_dataloader_rl: Optional[DataLoader],
                 eval_dataloader: DataLoader,
                 ref_policy_model: Policy,
                 policy_model: Policy,
                 value_model: Value,
                 reward_model: Reward,
                 optimizer: torch.optim.Optimizer,
                 init_step: int,
                 eval_accs: Dict,
                ):
        self.args = args
        self.train_dataloader = train_dataloader
        self.train_dataloader_qa = train_dataloader_qa
        self.train_dataloader_rl = train_dataloader_rl
        self.eval_dataloader = eval_dataloader
        self.ref_policy_model = ref_policy_model
        self.policy_model = policy_model
        self.value_model = value_model
        self.reward_model = reward_model
        self.optimizer = optimizer

        if self.args.mode == 'train':
            if not args.nolog and accelerator.is_main_process:
                wandb.init(project='rainier_stageII', name=args.run_name, config=args)
                wandb.define_metric('train/step')
                wandb.define_metric('eval/step')
                wandb.define_metric('train/*', step_metric='train/step')
                wandb.define_metric('eval/*', step_metric='eval/step', summary='max')

            if not self.args.half_half:
                self.train_sampler = iter(self.train_dataloader)
                for _ in range(init_step % len(self.train_dataloader)):
                    next(self.train_sampler)
            else:
                self.train_sampler_qa = iter(self.train_dataloader_qa)
                for _ in range(int(np.ceil(init_step / 2)) % len(self.train_dataloader_qa)):
                    next(self.train_sampler_qa)
                self.train_sampler_rl = iter(self.train_dataloader_rl)
                for _ in range(int(np.floor(init_step / 2)) % len(self.train_dataloader_rl)):
                    next(self.train_sampler_rl)

            self.eval_accs = eval_accs

        elif self.args.mode == 'eval':
            if not args.nolog and accelerator.is_main_process:
                wandb.init(project='rainier_eval', name=args.run_name, config=args)
                wandb.define_metric('eval/step')
                wandb.define_metric('eval/*', step_metric='eval/step')

    def loss(self, results):
        old_values = results['knowledges_value']
        old_logprobs = results['knowledges_logprobs']
        rewards = results['rewards/penalized']
        mask = results['knowledges_attention_mask'] # (B, KL)

        all_mask = accelerator.gather(mask) # (num_gpus * B, KL)
        weight = mask.sum(dim=1).float().mean().item() / all_mask.sum(dim=1).float().mean().item()

        with torch.no_grad():
            if self.args.whiten_rewards:
                whitened_rewards = whiten(rewards, mask, shift_mean=False, accelerator=accelerator)

            lastgaelam = 0
            advantages_reversed = []
            # gen_length = whitened_rewards.size(1)
            gen_length = mask.sum(dim=1).max().item() # to match the original implementation in V1
            for t in reversed(range(gen_length)):
                nextvalues = old_values[:, t + 1] if t < gen_length - 1 else 0.0
                delta = whitened_rewards[:, t] + self.args.gamma * nextvalues - old_values[:, t]
                lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], dim=1)
            advantages = F.pad(advantages, (0, whitened_rewards.size(1) - gen_length), value=0.0)
            returns = advantages + old_values

            whitened_advantages = whiten(advantages, mask, accelerator=accelerator).detach()

        forward_inputs = {
            'questions_input_ids': results['questions_input_ids'],
            'questions_attention_mask': results['questions_attention_mask'],
            'knowledges_input_ids': results['knowledges_input_ids'],
            'knowledges_attention_mask': results['knowledges_attention_mask'],
            'prefixed_knowledges_input_ids': results['prefixed_knowledges_input_ids'],
            'prefixed_knowledges_attention_mask': results['prefixed_knowledges_attention_mask'],
        }

        policy_forward = self.policy_model.forward_pass(**forward_inputs)
        new_logprobs = policy_forward['knowledges_logprobs']

        ratio = torch.exp(new_logprobs - old_logprobs)
        pg_losses1 = -whitened_advantages * ratio
        pg_losses2 = -whitened_advantages * torch.clamp(ratio, min=1.0 - self.args.cliprange, max=1.0 + self.args.cliprange)
        pg_loss = reduce_mean(torch.max(pg_losses1, pg_losses2), mask)
        pg_loss = pg_loss * weight

        if self.args.policy_value_sharing:
            new_values = policy_forward['knowledges_value']
        else:
            value_forward = self.value_model.forward_pass(**forward_inputs)
            new_values = value_forward['knowledges_value']
            new_values *= mask  # TODO: I doubt if this line is necessary

        new_values_clipped = clamp(new_values, old_values - self.args.cliprange_value, old_values + self.args.cliprange_value)
        vf_losses1 = torch.square(new_values - returns)
        vf_losses2 = torch.square(new_values_clipped - returns)
        vf_loss = .5 * reduce_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_loss = vf_loss * weight

        loss = self.args.pg_coef * pg_loss + self.args.vf_coef * vf_loss

        results['loss/total'] = loss
        results['loss/policy'] = pg_loss
        results['loss/value'] = vf_loss

    def train(self, step):
        self.save(step=step)
        self.valid(step=step)

        accelerator.wait_for_everyone()
        if not self.args.half_half:
            try:
                batch = next(self.train_sampler)
            except StopIteration:
                self.train_sampler = iter(self.train_dataloader)
                batch = next(self.train_sampler)
        else:
            if step % 2 == 0:
                try:
                    batch = next(self.train_sampler_qa)
                except StopIteration:
                    self.train_sampler_qa = iter(self.train_dataloader_qa)
                    batch = next(self.train_sampler_qa)
            else:
                try:
                    batch = next(self.train_sampler_rl)
                except StopIteration:
                    self.train_sampler_rl = iter(self.train_dataloader_rl)
                    batch = next(self.train_sampler_rl)

        results = copy.deepcopy(batch)

        # Rollout from current policy
        with torch.no_grad():
            rollouts = self.policy_model.sample(
                questions_input_ids=results['questions_input_ids'],
                questions_attention_mask=results['questions_attention_mask'],
                temperature=self.args.temperature,
            )
            results.update(rollouts)

        forward_inputs = {
            'questions_input_ids': results['questions_input_ids'],
            'questions_attention_mask': results['questions_attention_mask'],
            'knowledges_input_ids': results['knowledges_input_ids'],
            'knowledges_attention_mask': results['knowledges_attention_mask'],
            'prefixed_knowledges_input_ids': results['prefixed_knowledges_input_ids'],
            'prefixed_knowledges_attention_mask': results['prefixed_knowledges_attention_mask'],
        }

        with torch.no_grad():
            policy_forward = self.policy_model.forward_pass(**forward_inputs)
            results.update(policy_forward)

        # Run value network
        if not self.args.policy_value_sharing:
            with torch.no_grad(): # treat the values at beginning of step as ground-truth
                value_forward = self.value_model.forward_pass(**forward_inputs)
                results['knowledges_value'] = value_forward['knowledges_value']
                results['knowledges_value'] *= results['knowledges_attention_mask']  # TODO: I doubt if this line is necessary

        # Run ref policy
        with torch.no_grad():
            ref_policy_forward = self.ref_policy_model.forward_pass(**forward_inputs)
            results['knowledges_ref_logits'] = ref_policy_forward['knowledges_logits']
            results['knowledges_ref_logprobs'] = ref_policy_forward['knowledges_logprobs']

        # Get reward
        reward_results = self.reward_model.get_reward(
            questions_input_ids=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'questions_input_ids'],
            questions_attention_mask=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'questions_attention_mask'],
            choicess_input_ids=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'choicess_input_ids'],
            choicess_attention_mask=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'choicess_attention_mask'],
            choicess_labels=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'choicess_labels'],
            answers_labels=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'answers_labels'],
            answer_ixs=results['answer_ixs'],
            knowledges_input_ids=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'knowledges_input_ids'],
            knowledges_attention_mask=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'knowledges_attention_mask'],
        )
        results.update(reward_results)
        with torch.no_grad():
            self.reward_model.kl_penalize_reward(results)

        # Train
        if not self.args.half_half or (self.args.half_half and step % 2 == 0):
            self.optimizer.zero_grad()
            loss = self.args.qa_coef * results['loss/qa'] + self.args.qka_coef * results['loss/qka']
            accelerator.backward(loss)
            self.optimizer.step()

            loss_qa = results['loss/qa'].unsqueeze(0) # (1)
            loss_qka = results['loss/qka'].unsqueeze(0) # (1)
            corrects_knowless = accelerator.gather(results['corrects_knowless']) # (num_gpus * B)
            losses_qa = accelerator.gather(loss_qa) # (num_gpus)
            losses_qka = accelerator.gather(loss_qka) # (num_gpus)
            acc_knowless = corrects_knowless.float().mean().item()
            loss_qa = losses_qa.mean().item()
            loss_qka = losses_qka.mean().item()
            if not self.args.nolog and accelerator.is_main_process:
                if step % self.args.log_interval == 0:
                    wandb.log({
                        'train/step': step,
                        'train/acc_knowless': acc_knowless,
                        'train/loss/qa': loss_qa,
                        'train/loss/qka': loss_qka,
                    })
        if not self.args.half_half or (self.args.half_half and step % 2 == 1):
            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(self.args.noptepochs):
                self.optimizer.zero_grad()
                self.loss(results)
                accelerator.backward(results['loss/total'])
                if self.args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        chain(self.policy_model.model.parameters(),
                              self.value_model.model.parameters()),
                        self.args.max_grad_norm)
                self.optimizer.step()

            loss_total = results['loss/total'].unsqueeze(0) # (1)
            loss_policy = results['loss/policy'].unsqueeze(0) # (1)
            loss_value = results['loss/value'].unsqueeze(0) # (1)
            loss_qa = results['loss/qa'].unsqueeze(0) # (1)
            loss_qka = results['loss/qka'].unsqueeze(0) # (1)
            reward_penalized = torch.mean(reduce_sum(results['rewards/penalized'], results['knowledges_attention_mask'], axis=1)).unsqueeze(0) # (1)
            reward_kl = torch.mean(reduce_sum(results['rewards/kl'], results['knowledges_attention_mask'], axis=1)).unsqueeze(0) # (1)
            reward_normalized = results['rewards/normalized'].mean(dim=0, keepdim=True) # (1)
            reward_raw = results['rewards/raw'].mean(dim=0, keepdim=True) # (1)

            corrects = accelerator.gather(results['corrects']) # (num_gpus * B)
            corrects_knowless = accelerator.gather(results['corrects_knowless']) # (num_gpus * B)
            losses_total = accelerator.gather(loss_total) # (num_gpus)
            losses_policy = accelerator.gather(loss_policy) # (num_gpus)
            losses_value = accelerator.gather(loss_value) # (num_gpus)
            losses_qa = accelerator.gather(loss_qa) # (num_gpus)
            losses_qka = accelerator.gather(loss_qka) # (num_gpus)
            rewards_penalized = accelerator.gather(reward_penalized) # (num_gpus)
            rewards_kl = accelerator.gather(reward_kl) # (num_gpus)
            rewards_normalized = accelerator.gather(reward_normalized) # (num_gpus)
            rewards_raw = accelerator.gather(reward_raw) # (num_gpus)

            acc = corrects.float().mean().item()
            acc_knowless = corrects_knowless.float().mean().item()
            acc_delta = acc - acc_knowless
            loss_total = losses_total.mean().item()
            loss_policy = losses_policy.mean().item()
            loss_value = losses_value.mean().item()
            loss_qa = losses_qa.mean().item()
            loss_qka = losses_qka.mean().item()
            reward_penalized = rewards_penalized.mean().item()
            reward_kl = rewards_kl.mean().item()
            reward_normalized = rewards_normalized.mean().item()
            reward_raw = rewards_raw.mean().item()

            # Logging
            if not self.args.nolog and accelerator.is_main_process:
                if step % self.args.log_interval == 0:
                    wandb.log({
                        'train/step': step,
                        'train/acc': acc,
                        'train/acc_knowless': acc_knowless,
                        'train/acc_delta': acc_delta,
                        'train/loss/total': loss_total,
                        'train/loss/policy': loss_policy,
                        'train/loss/value': loss_value,
                        'train/loss/qa': loss_qa,
                        'train/loss/qka': loss_qka,
                        'train/reward/penalized': reward_penalized,
                        'train/reward/KL': reward_kl,
                        'train/reward/normalized': reward_normalized,
                        'train/reward/raw': reward_raw,
                    })

    def valid(self, step):
        if self.args.eval_loop_cap is not None and self.args.eval_loop_cap == 0:
            return
        if step % self.args.eval_interval != 0:
            return
        if step in self.eval_accs:
            return
        log_info(f'Evaluating [step {step}] ...')

        accelerator.wait_for_everyone()

        with torch.no_grad():
            corrects, corrects_knowless, task_ixs = [], [], []
            results_table = wandb.Table(columns=['step', 'id', 'task', 'question', 'knowledge', 'answer_ix', 'pred', 'correct', 'pred_knowless', 'correct_knowless', 'rectified'])
            for i, batch in enumerate(tqdm(self.eval_dataloader) if accelerator.is_main_process else self.eval_dataloader):
                if self.args.eval_loop_cap is not None and i == self.args.eval_loop_cap:
                    break

                results = copy.deepcopy(batch)

                rollouts = self.policy_model.sample(
                    questions_input_ids=results['questions_input_ids'],
                    questions_attention_mask=results['questions_attention_mask'],
                    sample=False,
                )
                results.update(rollouts)

                reward_results = self.reward_model.get_reward(
                    questions_input_ids=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'questions_input_ids'],
                    questions_attention_mask=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'questions_attention_mask'],
                    choicess_input_ids=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'choicess_input_ids'],
                    choicess_attention_mask=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'choicess_attention_mask'],
                    choicess_labels=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'choicess_labels'],
                    answers_labels=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'answers_labels'],
                    answer_ixs=results['answer_ixs'],
                    knowledges_input_ids=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'knowledges_input_ids'],
                    knowledges_attention_mask=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'knowledges_attention_mask'],
                    override_bias=0,
                    override_gain=1,
                )
                results.update(reward_results)

                corrects.append(results['corrects'])
                corrects_knowless.append(results['corrects_knowless'])
                task_ixs.append(results['task_ixs'])

                if accelerator.is_main_process:
                    results_table.add_data(step, i, self.eval_dataloader.dataset.tasks[results['task_ixs'][0]],
                                           results['questions_text'][0], results['knowledges_text'][0], results['answer_ixs'][0].item(),
                                           results['preds'][0].item(), results['corrects'][0].item(), results['preds_knowless'][0].item(), results['corrects_knowless'][0].item(), results['rectifieds'][0].item())

        corrects = torch.stack(corrects, dim=0) # (M, B)
        corrects_knowless = torch.stack(corrects_knowless, dim=0) # (M, B)
        task_ixs = torch.stack(task_ixs, dim=0) # (M, B)

        corrects = accelerator.gather(corrects.unsqueeze(0)) # (num_gpus, M, B)
        corrects_knowless = accelerator.gather(corrects_knowless.unsqueeze(0)) # (num_gpus, M, B)
        task_ixs = accelerator.gather(task_ixs.unsqueeze(0)) # (num_gpus, M, B)

        # Accelerator may pad the tensors to make them divisible by the total batch size
        corrects = corrects.transpose(0, 1).flatten(0, 2)[:len(self.eval_dataloader.dataset)] # (N)
        corrects_knowless = corrects_knowless.transpose(0, 1).flatten(0, 2)[:len(self.eval_dataloader.dataset)] # (N)
        task_ixs = task_ixs.transpose(0, 1).flatten(0, 2)[:len(self.eval_dataloader.dataset)] # (N)

        acc_weighted = corrects.float().mean().item()
        corrects_by_task = defaultdict(list)
        for task_ix, correct in zip(task_ixs, corrects):
            task = self.eval_dataloader.dataset.tasks[task_ix]
            corrects_by_task[task].append(correct)
        corrects_by_task = {k: torch.stack(v, dim=0) for k, v in corrects_by_task.items()}
        acc_by_task = {k: v.float().mean().item() for k, v in corrects_by_task.items()}
        acc_unweighted = np.mean(list(acc_by_task.values()))
        corrects_knowless_by_task = defaultdict(list)
        for task_ix, correct in zip(task_ixs, corrects_knowless):
            task = self.eval_dataloader.dataset.tasks[task_ix]
            corrects_knowless_by_task[task].append(correct)
        corrects_knowless_by_task = {k: torch.stack(v, dim=0) for k, v in corrects_knowless_by_task.items()}
        acc_knowless_by_task = {k: v.float().mean().item() for k, v in corrects_knowless_by_task.items()}
        acc_unweighted_knowless = np.mean(list(acc_knowless_by_task.values()))
        acc_unweighted_delta = acc_unweighted - acc_unweighted_knowless

        log_info(f'Evaluated [step {step}] acc_weighted = {acc_weighted:.4f} | acc_unweighted = {acc_unweighted:.4f}')
        log_info('Accuracy by task:')
        for task, acc in acc_by_task.items():
            log_info(f'\t{task} = {acc:.4f}')

        if not self.args.nolog and accelerator.is_main_process:
            stats = {
                'eval/step': step,
                'eval/results_table': results_table,
                'eval/acc_weighted': acc_weighted,
                'eval/acc_unweighted': acc_unweighted,
                'eval/acc_unweighted_knowless': acc_unweighted_knowless,
                'eval/acc_unweighted_delta': acc_unweighted_delta,
            }
            for task, acc in acc_by_task.items():
                stats[f'eval/acc/{task}'] = acc
            wandb.log(stats)

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

    def eval(self, step): # step=-1 for baseline
        if self.args.eval_loop_cap is not None and self.args.eval_loop_cap == 0:
            return
        log_info(f'Evaluating [step {step}] ...')

        accelerator.wait_for_everyone()

        with torch.no_grad():
            corrects, task_ixs = [], []
            results_table = wandb.Table(columns=['step', 'id', 'task', 'question', 'knowledge', 'answer_ix', 'pred', 'correct', 'pred_knowless', 'correct_knowless', 'rectified'])
            knowledge_outputs = []
            inference_outputs = []
            for i, batch in enumerate(tqdm(self.eval_dataloader) if accelerator.is_main_process else self.eval_dataloader):
                if self.args.eval_loop_cap is not None and i == self.args.eval_loop_cap:
                    break

                results = copy.deepcopy(batch)

                knowledgess_text, knowledgess_input_ids, knowledgess_attention_mask = [], [], [] # [K * (B, KL)]
                if step != -1: # If not baseline, generate knowledge
                    if 'knowledgess_input_ids' not in results:
                        if self.args.use_mcts:
                            from mcts import BatchedMCTS
                            MCTS = BatchedMCTS(self.policy_model.tokenizer, self.policy_model, self.value_model, ref_policy=self.ref_policy_model, reward_model=self.reward_model,
                                            batch_size=self.args.batch_size * 1, response_len=self.policy_model.tokenizer.max_knowledge_len, num_simulations=10, num_sparse_actions=2,
                                            kl_coef=0.0, # LJC: positive KL coef is not supported yet
                                            sample=False, topp=1.0,
                                            disable_cache=True, is_seq2seq=True, # LJC: cache is not supported yet for seq2seq model
                                            )
                            input_ids, attention_mask = results['questions_input_ids'], results['questions_attention_mask']
                            output_ids, output_mask = MCTS.generate(input_ids, attention_mask)
                            response_ids = output_ids[:, input_ids.size(1):] # (B * 1, RL)
                            response_mask = output_mask[:, input_ids.size(1):]
                            knowledges_text = self.policy_model.tokenizer.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) # (B * 1)
                            knowledges_text = [knowledge.lower() for knowledge in knowledges_text]
                            knowledgess_text.append(knowledges_text)
                            knowledges_tok = self.policy_model.tokenizer(knowledges_text, return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.policy_model.tokenizer.max_knowledge_len)
                            knowledges_input_ids, knowledges_attention_mask = knowledges_tok['input_ids'].to(response_ids.device), knowledges_tok['attention_mask'].to(response_mask.device)
                            knowledgess_input_ids, knowledgess_attention_mask = knowledges_input_ids.unsqueeze(0), knowledges_attention_mask.unsqueeze(0) # (1, B, KL)
                        else:
                            for j in range(self.args.num_samples):
                                rollouts = self.policy_model.sample(
                                    questions_input_ids=results['questions_input_ids'],
                                    questions_attention_mask=results['questions_attention_mask'],
                                    top_p=self.args.top_p,
                                )
                                knowledgess_text.append(rollouts[('' if self.args.do_not_lowercase else 'lowercased_') + 'knowledges_text'])
                                knowledgess_input_ids.append(rollouts[('' if self.args.do_not_lowercase else 'lowercased_') + 'knowledges_input_ids'])
                                knowledgess_attention_mask.append(rollouts[('' if self.args.do_not_lowercase else 'lowercased_') + 'knowledges_attention_mask'])
                            knowledgess_input_ids = torch.stack(knowledgess_input_ids, dim=0) # (K, B, KL)
                            knowledgess_attention_mask = torch.stack(knowledgess_attention_mask, dim=0) # (K, B, KL)
                    else:
                        knowledgess_text = results[('' if self.args.do_not_lowercase else 'lowercased_') + 'knowledgess_text']
                        knowledgess_input_ids = results[('' if self.args.do_not_lowercase else 'lowercased_') + 'knowledgess_input_ids']
                        knowledgess_attention_mask = results[('' if self.args.do_not_lowercase else 'lowercased_') + 'knowledgess_attention_mask']

                reward_results = self.reward_model.get_reward_ensemble(
                    questions_input_ids=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'questions_input_ids'],
                    questions_attention_mask=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'questions_attention_mask'],
                    choicess_input_ids=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'choicess_input_ids'],
                    choicess_attention_mask=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'choicess_attention_mask'],
                    choicess_labels=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'choicess_labels'],
                    answers_labels=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'answers_labels'],
                    answer_ixs=results['answer_ixs'],
                    knowledgess_input_ids=knowledgess_input_ids,
                    knowledgess_attention_mask=knowledgess_attention_mask,
                    override_bias=0,
                    override_gain=1,
                )
                results.update(reward_results)

                corrects.append(results['corrects'])
                task_ixs.append(results['task_ixs'])

                if accelerator.is_main_process:
                    selected_knowledge_ix = results['selected_knowledge_ixs'][0].item()
                    selected_knowledge = knowledgess_text[selected_knowledge_ix][0] if selected_knowledge_ix != -1 else ''
                    results_table.add_data(step, i, self.eval_dataloader.dataset.tasks[results['task_ixs'][0].item()],
                                           results['questions_text'][0], selected_knowledge, results['answer_ixs'][0].item(),
                                           results['preds'][0].item(), results['corrects'][0].item(), results['preds_knowless'][0].item(), results['corrects_knowless'][0].item(), results['rectifieds'][0].item())

                # TODO: Gather these results across all processes
                knowledgess_text_transposed = [list(x) for x in zip(*knowledgess_text)] if len(knowledgess_text) > 0 else [[] for _ in range(len(results['questions_text']))]
                for i, (task_ix, question, choices, answer_ix, knowledges) in enumerate(zip(results['task_ixs'], results['questions_text'], results['choicess_text'], results['answer_ixs'], knowledgess_text_transposed)):
                    choices = [choice for choice in choices if choice != ''] # remove padding
                    knowledges = [knowledge for knowledge in knowledges if knowledge != ''] # remove padding
                    item = {
                        'task': self.eval_dataloader.dataset.tasks[task_ix.item()],
                        'split': self.args.eval_split,
                        'query': question,
                        'cands': choices,
                        'answer_ix': answer_ix.item(),
                        'answer': choices[answer_ix.item()],
                        'knowledges': knowledges,
                    }
                    knowledge_outputs.append(copy.deepcopy(item))
                    item.update({
                        'scores_': results['answer_logitsss'][:1+len(knowledges), i, :len(choices)].tolist(),
                        'probs_': results['answer_probsss'][:1+len(knowledges), i, :len(choices)].tolist(),
                        'probs': results['answer_probss'][i, :len(choices)].tolist(),
                        'pred': results['preds'][i].item(),
                        'ok': int(results['corrects'][i].item()),
                        'pred_knowless': results['preds_knowless'][i].item(),
                        'ok_knowless': int(results['corrects_knowless'][i].item()),
                        'rectified': results['rectifieds'][i].item(),
                    })
                    inference_outputs.append(copy.deepcopy(item))

        corrects = torch.stack(corrects, dim=0) # (M, B)
        task_ixs = torch.stack(task_ixs, dim=0) # (M, B)

        corrects = accelerator.gather(corrects.unsqueeze(0)) # (num_gpus, M, B)
        task_ixs = accelerator.gather(task_ixs.unsqueeze(0)) # (num_gpus, M, B)

        # Accelerator may pad the tensors to make them divisible by the total batch size
        corrects = corrects.transpose(0, 1).flatten(0, 2)[:len(self.eval_dataloader.dataset)] # (N)
        task_ixs = task_ixs.transpose(0, 1).flatten(0, 2)[:len(self.eval_dataloader.dataset)] # (N)

        acc_weighted = corrects.float().mean().item()
        corrects_by_task = defaultdict(list)
        for task_ix, correct in zip(task_ixs, corrects):
            task = self.eval_dataloader.dataset.tasks[task_ix]
            corrects_by_task[task].append(correct)
        corrects_by_task = {k: torch.stack(v, dim=0) for k, v in corrects_by_task.items()}
        acc_by_task = {k: v.float().mean().item() for k, v in corrects_by_task.items()}
        acc_unweighted = np.mean(list(acc_by_task.values()))

        log_info(f'Evaluated [step {step}] acc_weighted = {acc_weighted:.4f} | acc_unweighted = {acc_unweighted:.4f}')
        log_info('Accuracy by task:')
        for task, acc in acc_by_task.items():
            log_info(f'\t{task} = {acc:.4f}')

        if not self.args.nolog and accelerator.is_main_process:
            stats = {
                'eval/step': step,
                'eval/results_table': results_table,
                'eval/acc_weighted': acc_weighted,
                'eval/acc_unweighted': acc_unweighted,
            }
            for task, acc in acc_by_task.items():
                stats[f'eval/acc/{task}'] = acc
            wandb.log(stats)

        if not self.args.nosave:
            def merge(path):
                paths = [f'{path}.{i}' for i in range(accelerator.num_processes)]
                outputss = []
                for p in paths:
                    with open(p, 'r') as f:
                        outputss.append(json.load(f))
                B = self.args.batch_size
                outputsss = [[outputs[i:i+B] for i in range(0, len(outputs), B)] for outputs in outputss] # (num_gpus, M, B)
                outputsss = list(map(list, zip(*outputsss))) # (M, num_gpus, B)
                outputss = list(chain(*outputsss)) # (M * num_gpus, B)
                outputs = list(chain(*outputss)) # (M * num_gpus * B)
                outputs = outputs[:len(self.eval_dataloader.dataset)]
                with open(path, 'w') as f:
                    json.dump(outputs, f, indent=4)
                for path in paths:
                    os.remove(path)
            if self.args.data_path.endswith('.json') and not ('{task}' in self.args.data_path and '{split}' in self.args.data_path):
                inference_path = os.path.join(self.args.inference_dir, f'inference_rainier-ckp{step}.{self.args.data_path.split("/")[-1]}.{accelerator.process_index}') \
                                if self.args.policy_reward_sharing else \
                                os.path.join(self.args.inference_dir, f'inference_{self.args.qa_model_type.split("/")[-1]}.{self.args.data_path.split("/")[-1]}.{accelerator.process_index}')
                with open(inference_path, 'w') as f:
                    json.dump(inference_outputs, f, indent=4)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    inference_path = os.path.join(self.args.inference_dir, f'inference_rainier-ckp{step}.{self.args.data_path.split("/")[-1]}') \
                                    if self.args.policy_reward_sharing else \
                                    os.path.join(self.args.inference_dir, f'inference_{self.args.qa_model_type.split("/")[-1]}.{self.args.data_path.split("/")[-1]}')
                    merge(inference_path)
            else:
                knowledge_path = os.path.join(self.args.knowledge_dir, f'knowledge_rainier-ckp{step}.json.{accelerator.process_index}')
                inference_path = os.path.join(self.args.inference_dir, f'inference_rainier-ckp{step}.knowledge_rainier-ckp{step}.json.{accelerator.process_index}') \
                                if self.args.policy_reward_sharing else \
                                os.path.join(self.args.inference_dir, f'inference_{self.args.qa_model_type.split("/")[-1]}.knowledge_rainier-ckp{step}.json.{accelerator.process_index}')
                with open(knowledge_path, 'w') as f:
                    json.dump(knowledge_outputs, f, indent=4)
                with open(inference_path, 'w') as f:
                    json.dump(inference_outputs, f, indent=4)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    knowledge_path = os.path.join(self.args.knowledge_dir, f'knowledge_rainier-ckp{step}.json')
                    inference_path = os.path.join(self.args.inference_dir, f'inference_rainier-ckp{step}.knowledge_rainier-ckp{step}.json') \
                                    if self.args.policy_reward_sharing else \
                                    os.path.join(self.args.inference_dir, f'inference_{self.args.qa_model_type.split("/")[-1]}.knowledge_rainier-ckp{step}.json')
                    merge(knowledge_path)
                    merge(inference_path)

    """
    Internally set bias and gain terms based on the data from the dataloader
    """
    def set_reward_norm(self):
        accelerator.wait_for_everyone()

        with torch.no_grad():
            rewards = []
            dataloader = self.train_dataloader_rl if self.args.half_half else self.train_dataloader
            for i, batch in enumerate(tqdm(dataloader) if accelerator.is_main_process else dataloader):
                if self.args.eval_loop_cap is not None and i == self.args.eval_loop_cap:
                    break
                results = copy.deepcopy(batch)
                rollouts = self.policy_model.sample(
                    questions_input_ids=results['questions_input_ids'],
                    questions_attention_mask=results['questions_attention_mask'],
                    temperature=self.args.temperature,
                )
                results.update(rollouts)
                reward_results = self.reward_model.get_reward(
                    questions_input_ids=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'questions_input_ids'],
                    questions_attention_mask=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'questions_attention_mask'],
                    choicess_input_ids=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'choicess_input_ids'],
                    choicess_attention_mask=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'choicess_attention_mask'],
                    choicess_labels=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'choicess_labels'],
                    answers_labels=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'answers_labels'],
                    answer_ixs=results['answer_ixs'],
                    knowledges_input_ids=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'knowledges_input_ids'],
                    knowledges_attention_mask=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'knowledges_attention_mask'],
                    override_bias=0,
                    override_gain=1,
                )
                results.update(reward_results)
                rewards.append(results['rewards/raw'])

        rewards = torch.stack(rewards, dim=0) # (M, B)
        rewards = accelerator.gather(rewards.unsqueeze(0)) # (num_gpus, M, B)
        rewards = rewards.transpose(0, 1).flatten(0, 2)[:len(dataloader.dataset)] # (N)

        old_mean, old_std = rewards.mean().item(), rewards.std().item()
        new_mean, new_std = 0.0, 1.0
        self.reward_model.gain = new_std / old_std
        self.reward_model.bias = new_mean - self.reward_model.gain * old_mean

        log_info(f'Reward normalization coefficients set to: gain = {self.reward_model.gain:.4f} | bias = {self.reward_model.bias:.4f}')

    def save(self, step):
        if self.args.nosave:
            return
        if step % self.args.save_interval != 0:
            return
        # this will overwrite an existing ckpt with the save filename!
        accelerator.wait_for_everyone()
        if accelerator.distributed_type == accelerate.utils.DistributedType.FSDP:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.policy_model.model, StateDictType.FULL_STATE_DICT, save_policy):
                policy_model_state_dict = self.policy_model.model.state_dict()
            with FSDP.state_dict_type(self.policy_model.linear, StateDictType.FULL_STATE_DICT, save_policy):
                policy_linear_state_dict = self.policy_model.linear.state_dict()
            if not self.args.policy_value_sharing:
                with FSDP.state_dict_type(self.value_model.model, StateDictType.FULL_STATE_DICT, save_policy):
                    value_model_state_dict = self.value_model.model.state_dict()
        else:
            policy_model_state_dict = accelerator.unwrap_model(self.policy_model.model).state_dict()
            policy_linear_state_dict = accelerator.unwrap_model(self.policy_model.linear).state_dict()
            if not self.args.policy_value_sharing:
                value_model_state_dict = accelerator.unwrap_model(self.value_model.model).state_dict()
        result = {
            'model': policy_model_state_dict,
            'linear': policy_linear_state_dict,
            'step': step,
            'eval_accs': self.eval_accs,
        }
        if not self.args.policy_value_sharing:
            result['value_model'] = value_model_state_dict
        accelerator.save(result, f'{self.args.model_dir}/last.pth')
        log_info(f'[step {step}] model checkpoint saved')


def main():
    args = get_args()

    set_seed(args.seed, args.cuda_deterministic)

    # Set up save directories
    if not args.nosave:
        if args.mode == 'train':
            args.output_dir = '../runs'
            if args.load_from_ckpt is not None:
                args.save_dir = os.path.dirname(os.path.dirname(args.load_from_ckpt))
                args.run_name = args.save_dir.split('/')[-1]
            else:
                args.save_dir = os.path.join(args.output_dir, args.run_name)
            args.reward_dir = os.path.join(args.save_dir, 'reward')
            args.model_dir = os.path.join(args.save_dir, 'model')
            args.knowledge_dir = os.path.join(args.save_dir, 'knowledge')
            args.inference_dir = os.path.join(args.save_dir, 'inference')
            if accelerator.is_main_process:
                for d in [args.save_dir, args.reward_dir, args.model_dir, args.knowledge_dir, args.inference_dir]:
                    ensure_dir(d)

        elif args.mode == 'eval':
            if args.load_from_ckpt is not None:
                args.save_dir = os.path.dirname(os.path.dirname(args.load_from_ckpt))
                args.save_dir = args.save_dir.replace('runs', 'eval')
                ckp = args.load_from_ckpt.split('ckp_')[-1].strip('.pth')
                args.save_dir += f'_ckp-{ckp}'
            elif args.eval_ckpt is not None:
                args.save_dir = os.path.dirname(args.eval_ckpt)
            else:
                log.error('You must provide either --ckpt or --load_from_ckpt!')
                exit(-1)
            args.run_name = args.save_dir.split('/')[-1]
            args.knowledge_dir = os.path.join(args.save_dir, 'knowledge')
            args.inference_dir = os.path.join(args.save_dir, 'inference')
            if accelerator.is_main_process:
                for d in [args.save_dir, args.knowledge_dir, args.inference_dir]:
                    ensure_dir(d)

        log_info(f'Write to output directory: {args.save_dir}')
        if accelerator.is_main_process:
            with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    # Load data
    log_info(f'Loading data ...')

    tokenizer = transformers.T5Tokenizer.from_pretrained(args.model_type)
    tokenizer.max_question_len = args.max_question_len
    tokenizer.max_answer_len = args.max_answer_len
    tokenizer.max_knowledge_len = args.max_knowledge_len

    if args.mode == 'train':
        if not args.half_half:
            train_dataset = QADataset(args, 'train', args.train_tasks, args.data_path, tokenizer)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=train_dataset.collate_fn)
            train_dataloader = accelerator.prepare(train_dataloader)
            train_dataset_qa, train_dataset_rl = None, None
            train_dataloader_qa, train_dataloader_rl = None, None
        else:
            train_dataset_qa = QADataset(args, 'train', args.train_tasks, args.data_path, tokenizer, 'qa')
            train_dataset_rl = QADataset(args, 'train', args.train_tasks, args.data_path, tokenizer, 'rl')
            train_dataloader_qa = DataLoader(train_dataset_qa, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=train_dataset_qa.collate_fn)
            train_dataloader_rl = DataLoader(train_dataset_rl, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=train_dataset_rl.collate_fn)
            train_dataloader_qa, train_dataloader_rl = accelerator.prepare(train_dataloader_qa, train_dataloader_rl)
            train_dataset = None
            train_dataloader = None
        eval_dataset = QADataset(args, 'dev', args.train_tasks, args.data_path, tokenizer)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=eval_dataset.collate_fn)
        eval_dataloader = accelerator.prepare(eval_dataloader)

    elif args.mode == 'eval':
        train_dataset = None
        train_dataloader = None
        train_dataset_qa = None
        train_dataset_rl = None
        train_dataloader_qa = None
        train_dataloader_rl = None
        eval_dataset = QADataset(args, args.eval_split, args.eval_tasks, args.data_path, tokenizer)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=eval_dataset.collate_fn)
        eval_dataloader = accelerator.prepare(eval_dataloader)

    # Initialize models and optimizer
    log_info(f'Initializing models ...')
    if args.mode == 'train':
        ref_policy = Policy(
            model_type=args.model_type,
            model_ckpt=args.model_ckpt,
            tokenizer=tokenizer,
            policy_value_sharing=args.policy_value_sharing,
            policy_reward_sharing=args.policy_reward_sharing,
            accelerator=accelerator,
        )
        ref_policy.model, ref_policy.linear = accelerator.prepare(ref_policy.model, ref_policy.linear)
        policy = Policy(
            model_type=args.model_type,
            model_ckpt=args.model_ckpt,
            tokenizer=tokenizer,
            policy_value_sharing=args.policy_value_sharing,
            policy_reward_sharing=args.policy_reward_sharing,
            accelerator=accelerator,
        )
        policy.model, policy.linear = accelerator.prepare(policy.model, policy.linear)
        value = Value(
            model_type=args.model_type,
            model_ckpt=args.model_ckpt if args.use_model_ckpt_for_value else None,
            model=policy.model if args.policy_value_sharing else None,
            tokenizer=tokenizer,
        )
        if not args.policy_value_sharing:
            value.model = accelerator.prepare(value.model)
        reward = Reward(
            model_type=args.qa_model_type,
            model_ckpt=args.qa_model_ckpt,
            model=policy.model if args.policy_reward_sharing else None,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            reward_shape=args.reward_shape,
            kl_coef=args.kl_coef,
            ensembling=args.ensembling,
            do_not_lowercase=args.do_not_lowercase,
        )
        if not args.policy_reward_sharing:
            reward.model = accelerator.prepare(reward.model)

        # We never need to optimize the reward model's parameters separately!
        if args.policy_value_sharing:
            parameters = chain(policy.model.parameters(), policy.linear.parameters())
        else:
            parameters = chain(policy.model.parameters(), policy.linear.parameters(), value.model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.lr, eps=1e-5)
        optimizer = accelerator.prepare(optimizer)
        args.total_steps = ceil_div(args.total_episodes, args.batch_size * int(os.environ['SLURM_GPUS_ON_NODE']) * int(os.environ['SLURM_JOB_NUM_NODES']))
        init_step = 0
        eval_accs = {}

        if args.load_from_stageI_ckpt is not None:
            checkpoint = torch.load(args.load_from_stageI_ckpt, map_location='cpu')
            accelerator.unwrap_model(policy.model).load_state_dict(checkpoint['model'])
            accelerator.unwrap_model(ref_policy.model).load_state_dict(checkpoint['model'])
            checkpoint.clear()

    elif args.mode == 'eval':
        ref_policy = Policy(
            model_type=args.model_type,
            model_ckpt=None,
            tokenizer=tokenizer,
            policy_value_sharing=args.policy_value_sharing,
            policy_reward_sharing=args.policy_reward_sharing,
            accelerator=accelerator,
        )
        ref_policy.model, ref_policy.linear = accelerator.prepare(ref_policy.model, ref_policy.linear)
        policy = Policy(
            model_type=args.model_type,
            model_ckpt=args.model_ckpt,
            tokenizer=tokenizer,
            policy_value_sharing=args.policy_value_sharing,
            policy_reward_sharing=args.policy_reward_sharing,
            accelerator=accelerator,
        )
        optimizer = None
        init_step = 0
        eval_accs = {}

        # checkpoint = None
        # if args.load_from_ckpt is not None:
        #     checkpoint = torch.load(args.load_from_ckpt, map_location='cpu')
        #     policy.model.load_state_dict(checkpoint['model'], strict=False)
        #     checkpoint.clear()
        # elif args.eval_ckpt is not None:
        #     checkpoint = torch.load(args.eval_ckpt, map_location='cpu')
        #     policy.model.load_state_dict(checkpoint, strict=False)
        #     checkpoint.clear()

        policy.model, policy.linear = accelerator.prepare(policy.model, policy.linear)

        value = Value(
            model_type=args.model_type,
            model_ckpt=args.model_ckpt + '-value',
            model=policy.model if args.policy_value_sharing else None,
            tokenizer=tokenizer,
        )
        if not args.policy_value_sharing:
            value.model = accelerator.prepare(value.model)
        reward = Reward(
            model_type=args.qa_model_type,
            model_ckpt=args.qa_model_ckpt,
            model=policy.model if args.policy_reward_sharing else None,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            reward_shape=args.reward_shape,
            kl_coef=args.kl_coef,
            ensembling=args.ensembling,
            do_not_lowercase=args.do_not_lowercase,
        )
        if not args.policy_reward_sharing:
            reward.model = accelerator.prepare(reward.model)

    # Set up trainer
    trainer = PPOTrainer(
        args=args,
        train_dataloader=train_dataloader,
        train_dataloader_qa=train_dataloader_qa,
        train_dataloader_rl=train_dataloader_rl,
        eval_dataloader=eval_dataloader,
        ref_policy_model=ref_policy,
        policy_model=policy,
        value_model=value,
        reward_model=reward,
        optimizer=optimizer,
        init_step=init_step,
        eval_accs=eval_accs,
    )

    # Normalize the rewards to so that initially they have mean 0, var 1
    if args.mode == 'train':
        if args.load_from_ckpt is None:
            log_info('Setting reward norm')
            if args.gain is not None and args.bias is not None:
                reward.gain = args.gain
                reward.bias = args.bias
            else:
                trainer.set_reward_norm()
            log_info(f'Set reward norm as gain = {reward.gain}, bias = {reward.bias}')
            if not args.nosave and accelerator.is_main_process:
                reward.write_reward_norm(args.reward_dir)

    # Evaluate baseline (no knowledge)
    if args.eval_baseline:
        trainer.eval(step=-1)

    # Train or evaluate
    if args.mode == 'train':
        steps = list(range(init_step, args.total_steps + 1))
        steps = tqdm(steps) if accelerator.is_main_process else steps
        for step in steps:
            trainer.train(step)
    elif args.mode == 'eval':
        trainer.eval(init_step)


if __name__ == '__main__':
    main()
