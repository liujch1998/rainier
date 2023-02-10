import argparse
from collections import defaultdict
from itertools import chain
import json
import numpy as np
import os
import random
import shutil
from tqdm import tqdm
from typing import Dict

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

from utils.utils import ensure_dir, set_seed, reduce_mean, reduce_sum, ceil_div, whiten, clamp
from args import get_args
from model.policy import Policy
from model.value import Value
from model.reward import Reward
from ppo import PPOTrainer

log = accelerate.logging.get_logger(__name__, log_level='INFO')


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
    'wsc273_': {'default': 'wsc273_'},
    'copa_': {'default': 'copa_'},
    'numersense_': {'default': 'numersense_'},
    'truthfulqa_mc1': {'default': 'truthfulqa_mc1'},
}

# This does not lowercase the data, by default
class QADataset(Dataset):
    def __init__(self, args, split, tasks, data_path, tokenizer):
        super().__init__()
        self.args = args
        self.split = split
        self.tasks = tasks.split(',')
        self.data_path = data_path
        self.tokenizer = tokenizer

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
                    for line in f:
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
                            'question': q,
                            'choices': choices,
                            'answer': a,
                            'answer_ix': answer_ix,
                        })
                log.info(f'Loaded dataset for task {task} split {self.split}, skipped {skipped} instances')
        elif self.data_path.endswith('.json'):
            for task_ix, task in enumerate(self.tasks):
                skipped = 0
                with open(self.data_path.replace('{task}', task).replace('{split}', self.split)) as f:
                    js = json.load(f)
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
                            'question': q,
                            'choices': choices,
                            'answer': a,
                            'answer_ix': answer_ix,
                            'knowledges': knowledges,
                        })
                log.info(f'Loaded dataset for task {task} split {self.split}, skipped {skipped} instances')
        log.info(f'Loaded split {self.split} with {len(instances)} total instances')
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

        padded_choicess = [item['choices'] + [''] * (8 - len(item['choices'])) for item in batch]
        choicess_tok = [self.tokenizer.batch_encode_plus(
            padded_choices,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.tokenizer.max_answer_len)
            for padded_choices in padded_choicess]
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

        lowercased_padded_choicess = [[choice.lower() for choice in choices] for choices in padded_choicess]
        lowercased_choicess_tok = [self.tokenizer.batch_encode_plus(
            choices,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.tokenizer.max_answer_len)
            for choices in lowercased_padded_choicess]
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
            'questions_input_ids': questions_input_ids,
            'questions_attention_mask': questions_attention_mask,
            'choicess_input_ids': choicess_input_ids,
            'choicess_attention_mask': choicess_attention_mask,
            'choicess_labels': choicess_labels,
            'answers_labels': answers_labels,
            'lowercased_choicess_input_ids': lowercased_choicess_input_ids,
            'lowercased_choicess_attention_mask': lowercased_choicess_attention_mask,
            'lowercased_choicess_labels': lowercased_choicess_labels,
            'lowercased_answers_labels': lowercased_answers_labels,
        }

        if 'knowledges' in batch[0]:
            knowledgess = [item['knowledges'] + [''] * (self.args.num_samples - len(item['knowledges'])) for item in batch]
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

            knowledgess_with_prefix = [[f'Knowledge: {k}' for k in ks] for ks in knowledgess]
            knowledgess_with_prefix_tok = [self.tokenizer.batch_encode_plus(
                knowledges_with_prefix,
                return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.tokenizer.max_knowledge_len)
                for knowledges_with_prefix in knowledgess_with_prefix]
            knowledgess_with_prefix_input_ids = torch.stack([knowledges_with_prefix_tok.input_ids for knowledges_with_prefix_tok in knowledgess_with_prefix_tok], dim=0)
            knowledgess_with_prefix_attention_mask = torch.stack([knowledges_with_prefix_tok.attention_mask for knowledges_with_prefix_tok in knowledgess_with_prefix_tok], dim=0)
            knowledgess_with_prefix_labels = knowledgess_with_prefix_input_ids.clone()
            knowledgess_with_prefix_labels[knowledgess_with_prefix_attention_mask == 0] = -100

            # questions_knowledgess = [[f'{q} \\n {k}' for k in ks] for q, ks in zip(questions, knowledgess)]
            # questions_knowledgess_tok = [self.tokenizer.batch_encode_plus(
            #     questions_knowledges,
            #     return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.tokenizer.max_question_len + self.tokenizer.max_knowledge_len)
            #     for questions_knowledges in questions_knowledgess]
            # questions_knowledgess_input_ids = torch.stack([questions_knowledges_tok.input_ids for questions_knowledges_tok in questions_knowledgess_tok], dim=0)
            # questions_knowledgess_attention_mask = torch.stack([questions_knowledges_tok.attention_mask for questions_knowledges_tok in questions_knowledgess_tok], dim=0)

            result.update({
                'knowledgess_input_ids': knowledgess_input_ids,
                'knowledegss_attention_mask': knowledgess_attention_mask,
                'knowledgess_labels': knowledgess_labels,
                'lowercased_knowledgess_input_ids': lowercased_knowledgess_input_ids,
                'lowercased_knowledgess_attention_mask': lowercased_knowledgess_attention_mask,
                'lowercased_knowledgess_labels': lowercased_knowledgess_labels,
                'knowledgess_with_prefix_input_ids': knowledgess_with_prefix_input_ids,
                'knowledegss_with_prefix_attention_mask': knowledgess_with_prefix_attention_mask,
                'knowledgess_with_prefix_labels': knowledgess_with_prefix_labels,
                # 'question_knowledgess_input_ids': questions_knowledgess_input_ids,
                # 'question_knowledgess_attention_mask': questions_knowledgess_attention_mask,
            })
        
        return result


class PPOTrainer:
    def __init__(self,
                 args: argparse.Namespace,
                 train_dataloader: DataLoader,
                 eval_dataloader: DataLoader,
                 ref_policy_model: Policy,
                 policy_model: Policy,
                 value_model: Value,
                 reward_model: Reward,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LambdaLR,
                 init_step: int,
                 eval_accs: Dict,
                 device,
                 accelerator,
                ):
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.ref_policy_model = ref_policy_model
        self.policy_model = policy_model
        self.value_model = value_model
        self.reward_model = reward_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.accelerator = accelerator

        if self.args.mode == 'train':
            if not args.nosave and self.accelerator.is_main_process:
                wandb.init(project='rainier_stageII', name=args.run_name, config=args)
                wandb.define_metric('train/step')
                wandb.define_metric('eval/step')
                wandb.define_metric('train/*', step_metric='train/step')
                wandb.define_metric('eval/*', step_metric='eval/step', summary='max')

            self.train_sampler = iter(self.train_dataloader)
            for _ in range(init_step % len(self.train_dataloader)):
                next(self.train_sampler)

            self.eval_accs = eval_accs

        elif self.args.mode == 'eval':
            if not args.nosave and self.accelerator.is_main_process:
                wandb.init(project='rainier_eval', name=args.run_name, config=args)
                wandb.define_metric('eval/step')
                wandb.define_metric('eval/*', step_metric='eval/step')

    def loss(self, results):
        old_values = results['knowledges_value']
        old_logprobs = results['knowledges_logprobs']
        rewards = results['rewards/penalized']
        mask = results['knowledges_attention_mask']

        with torch.no_grad():
            if self.args.whiten_rewards:
                rewards = whiten(rewards, mask, shift_mean=False)

            lastgaelam = 0
            advantages_reversed = []
            gen_length = rewards.size(1)
            for t in reversed(range(gen_length)):
                nextvalues = old_values[:, t + 1] if t < gen_length - 1 else 0.0
                delta = rewards[:, t] + self.args.gamma * nextvalues - old_values[:, t]
                lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], dim=1)
            returns = advantages + old_values

            advantages = whiten(advantages, mask).detach()

        forward_inputs = {
            'questions_input_ids': results['questions_input_ids'],
            'questions_attention_mask': results['questions_attention_mask'],
            'knowledges_input_ids': results['knowledges_input_ids'],
            'knowledges_attention_mask': results['knowledges_attention_mask'],
        }

        policy_forward = self.policy_model.forward_pass(**forward_inputs)
        new_logprobs = policy_forward['knowledges_logprobs']

        ratio = torch.exp(new_logprobs - old_logprobs)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, min=1.0 - self.args.cliprange, max=1.0 + self.args.cliprange)
        pg_loss = reduce_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = reduce_mean(torch.gt(pg_losses2, pg_losses).float(), mask)

        value_forward = self.value_model.forward_pass(**forward_inputs)
        new_values = value_forward['knowledges_value']
        new_values *= mask  # TODO: I doubt if this line is necessary

        new_values_clipped = clamp(new_values, old_values - self.args.cliprange_value, old_values + self.args.cliprange_value)
        vf_losses1 = torch.square(new_values - returns)
        vf_losses2 = torch.square(new_values_clipped - returns)
        vf_loss = .5 * reduce_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_clipfrac = reduce_mean(torch.gt(vf_losses2, vf_losses1).float(), mask)

        loss = self.args.pg_coef * pg_loss + self.args.vf_coef * vf_loss

        results['loss/total'] = loss
        results['loss/policy'] = pg_loss
        results['loss/value'] = vf_loss

    def train(self, step):
        self.save(step=step)
        self.valid(step=step)

        accelerate.utils.wait_for_everyone()
        try:
            batch = next(self.train_sampler)
        except StopIteration:
            self.train_sampler = iter(self.train_dataloader)
            batch = next(self.train_sampler)

        # Rollout from current policy
        with torch.no_grad():
            results = self.policy_model.sample(
                questions_input_ids=batch['questions_input_ids'],
                questions_attention_mask=batch['questions_attention_mask'],
                temperature=self.args.temperature,
            )

        forward_inputs = {
            'questions_input_ids': results['questions_input_ids'],
            'questions_attention_mask': results['questions_attention_mask'],
            'knowledges_input_ids': results['knowledges_input_ids'],
            'knowledges_attention_mask': results['knowledges_attention_mask'],
        }

        # Run value network
        with torch.no_grad(): # treat the values at beginning of step as ground-truth
            value_forward = self.value_model.forward_pass(**forward_inputs)
            results['knowledges_value'] = value_forward['knowledges_value']#.to(self.policy_model.device)
            results['knowledges_value'] *= results['knowledges_attention_mask']  # TODO: I doubt if this line is necessary

        # Run ref policy
        with torch.no_grad():
            ref_policy_forward = self.ref_policy_model.forward_pass(**forward_inputs)
            results['knowledges_ref_logits'] = ref_policy_forward['knowledges_logits']#.to(self.policy_model.device)
            results['knowledges_ref_logprobs'] = ref_policy_forward['knowledges_logprobs']#.to(self.policy_model.device)

        # Get reward
        with torch.no_grad():
            reward_results = self.reward_model.get_reward(
                questions_input_ids=batch['questions_input_ids'],
                questions_attention_mask=batch['questions_attention_mask'],
                choicess_input_ids=batch['choicess_input_ids'],
                choicess_attention_mask=batch['choicess_attention_mask'],
                choicess_labels=batch['choicess_labels'],
                answer_ixs=batch['answer_ixs'],
                knowledges_input_ids=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'knowledges_input_ids'],
                knowledges_attention_mask=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'knowledges_attention_mask'],
            )
            results = {**results, **reward_results}
            self.reward_model.kl_penalize_reward(results)

        # Train
        # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
        for ppo_epoch_idx in range(self.args.noptepochs):
            self.optimizer.zero_grad()
            self.loss(results)
            self.accelerator.backward(results['loss/total'])
            if self.args.clip_grad:
                torch.nn.utils.clip_grad_norm_(
                    chain(self.policy_model.model.parameters(),
                          self.value_model.model.parameters()),
                    self.args.max_grad_norm)
            self.optimizer.step()

        # Increment scheduler
        self.scheduler.step()

        loss_total = results['loss/total'].unsqueeze(0) # (1)
        loss_policy = results['loss/policy'].unsqueeze(0) # (1)
        loss_value = results['loss/value'].unsqueeze(0) # (1)
        reward_penalized = torch.mean(reduce_sum(results['rewards/penalized'], results['knowledges_attention_mask'], axis=1)).unsqueeze(0) # (1)
        reward_kl = torch.mean(reduce_sum(results['rewards/kl'], results['knowledges_attention_mask'], axis=1)).unsqueeze(0) # (1)
        reward_normalized = results['rewards/normalized'].mean(dim=0, keepdim=True) # (1)
        reward_raw = results['rewards/raw'].mean(dim=0, keepdim=True) # (1)

        corrects = self.accelerator.gather(results['corrects']) # (num_gpus * B)
        losses_total = self.accelerator.gather(loss_total) # (num_gpus)
        losses_policy = self.accelerator.gather(loss_policy) # (num_gpus)
        losses_value = self.accelerator.gather(loss_value) # (num_gpus)
        rewards_penalized = self.accelerator.gather(reward_penalized) # (num_gpus)
        rewards_kl = self.accelerator.gather(reward_kl) # (num_gpus)
        rewards_normalized = self.accelerator.gather(reward_normalized) # (num_gpus)
        rewards_raw = self.accelerator.gather(reward_raw) # (num_gpus)

        acc = corrects.float().mean().item()
        loss_total = losses_total.mean().item()
        loss_policy = losses_policy.mean().item()
        loss_value = losses_value.mean().item()
        reward_penalized = rewards_penalized.mean().item()
        reward_kl = rewards_kl.mean().item()
        reward_normalized = rewards_normalized.mean().item()
        reward_raw = rewards_raw.mean().item()

        # Logging
        if not self.args.nosave and self.accelerator.is_main_process:
            if step % self.args.log_interval == 0:
                wandb.log({
                    'train/step': step,
                    'train/acc': acc,
                    'train/loss/total': loss_total,
                    'train/loss/policy': loss_policy,
                    'train/loss/value': loss_value,
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
        log.info(f'Evaluating [step {step}] ...')

        accelerate.utils.wait_for_everyone()

        with torch.no_grad():
            corrects, task_ixs = [], []
            # results_table = wandb.Table(columns=['step', 'id', 'task', 'question', 'knowledge', 'pred', 'answer_ix', 'correct'])
            for i, batch in enumerate(tqdm(self.eval_dataloader)):
                if self.args.eval_loop_cap is not None and i == self.args.eval_loop_cap:
                    break

                results = self.policy_model.sample(
                    questions_input_ids=batch['questions_input_ids'],
                    questions_attention_mask=batch['questions_attention_mask'],
                    sample=False,
                )

                results = self.reward_model.get_reward(
                    questions_input_ids=batch['questions_input_ids'],
                    questions_attention_mask=batch['questions_attention_mask'],
                    choicess_input_ids=batch['choicess_input_ids'],
                    choicess_attention_mask=batch['choicess_attention_mask'],
                    choicess_labels=batch['choicess_labels'],
                    answer_ixs=batch['answer_ixs'],
                    knowledges_input_ids=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'knowledges_input_ids'],
                    knowledges_attention_mask=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'knowledges_attention_mask'],
                    override_bias=0,
                    override_gain=1,
                )

                corrects.append(results['corrects'])
                task_ixs.append(batch['task_ixs'])
                # results_table.add_data(step, i, batch['task'][0], batch['question'][0], knowledges[0], results['preds'][0], batch['answer_ix'][0], results['corrects'][0])

        corrects = torch.cat(corrects, dim=0) # (N)
        task_ixs = torch.cat(task_ixs, dim=0) # (N)

        corrects = self.accelerator.gather(corrects) # (num_gpus * N)
        task_ixs = self.accelerator.gather(task_ixs) # (num_gpus * N)

        # Accelerator may pad the tensors to make them divisible by the total batch size
        corrects = corrects[:len(self.eval_dataloader.dataset)]
        task_ixs = task_ixs[:len(self.eval_dataloader.dataset)]

        acc_weighted = corrects.float().mean().item()
        corrects_by_task = defaultdict(list)
        for task_ix, correct in zip(task_ixs, corrects):
            task = self.eval_dataloader.dataset.tasks[task_ix]
            corrects_by_task[task].append(correct)
        corrects_by_task = {k: torch.stack(v, dim=0) for k, v in corrects_by_task.items()}
        acc_by_task = {k: v.float().mean().item() for k, v in corrects_by_task.items()}
        acc_unweighted = np.mean(list(acc_by_task.values()))

        log.info(f'Evaluated [step {step}] acc_weighted = {acc_weighted:.4f} | acc_unweighted = {acc_unweighted:.4f}')
        log.info('Accuracy by task:')
        for task, acc in acc_by_task.items():
            log.info(f'\t{task} = {acc:.4f}')

        if not self.args.nosave and self.accelerator.is_main_process:
            stats = {
                'eval/step': step,
                # 'eval/results_table': results_table,
                'eval/acc_weighted': acc_weighted,
                'eval/acc_unweighted': acc_unweighted,
            }
            for task, acc in acc_by_task.items():
                stats[f'eval/acc/{task}'] = acc
            wandb.log(stats)

            prev_best_step = None if len(self.eval_accs) == 0 else max(self.eval_accs, key=self.eval_accs.get)
            self.eval_accs[step] = acc_unweighted
            if prev_best_step is None or acc_unweighted > self.eval_accs[prev_best_step]:
                if prev_best_step is not None:
                    try:
                        os.remove(f'{self.args.model_dir}/ckp_{prev_best_step}.pth')
                    except:
                        log.warning(f'Cannot remove previous best ckpt!')
                shutil.copy(f'{self.args.model_dir}/last.pth', f'{self.args.model_dir}/ckp_{step}.pth')
                log.info(f'Best ckpt updated to [step {step}]')
        else:
            self.eval_accs[step] = acc_unweighted

    def eval(self, step): # step=-1 for baseline
        log.info(f'Evaluating [step {step}] ...')

        accelerate.utils.wait_for_everyone()

        with torch.no_grad():
            corrects, task_ixs = [], []
            knowledge_outputs = []
            inference_outputs = []
            for i, batch in enumerate(tqdm(self.eval_dataloader)):
                knowledgess_input_ids, knowledgess_attention_mask = [], [] # [K * (B, KL)]
                if step != -1: # If not baseline, generate knowledge
                    if 'knowledges' not in batch:
                        for j in range(self.args.num_samples):
                            results = self.policy_model.sample(
                                questions_input_ids=batch['questions_input_ids'],
                                questions_attention_mask=batch['questions_attention_mask'],
                                top_p=self.args.top_p,
                            )
                            knowledgess_input_ids.append(results[('' if self.args.do_not_lowercase else 'lowercased_') + 'knowledges_input_ids'])
                            knowledgess_attention_mask.append(results[('' if self.args.do_not_lowercase else 'lowercased_') + 'knowledges_attention_mask'])
                        knowledgess_input_ids = torch.stack(knowledgess_input_ids, dim=0) # (K, B, KL)
                        knowledgess_attention_mask = torch.stack(knowledgess_attention_mask, dim=0) # (K, B, KL)
                    else:
                        knowledgess_input_ids = batch[('' if self.args.do_not_lowercase else 'lowercased_') + 'knowledgess_input_ids']
                        knowledgess_attention_mask = batch[('' if self.args.do_not_lowercase else 'lowercased_') + 'knowledgess_attention_mask']

                results = self.reward_model.get_reward(
                    questions_input_ids=batch['questions_input_ids'],
                    questions_attention_mask=batch['questions_attention_mask'],
                    choicess_input_ids=batch['choicess_input_ids'],
                    choicess_attention_mask=batch['choicess_attention_mask'],
                    choicess_labels=batch['choicess_labels'],
                    answer_ixs=batch['answer_ixs'],
                    knowledges_input_ids=knowledgess_input_ids,
                    knowledges_attention_mask=knowledgess_attention_mask,
                    override_bias=0,
                    override_gain=1,
                )

                corrects.append(results['corrects'])
                task_ixs.append(batch['task_ix'])

                # TODO: fix this!
                # knowledgess = [list(x) for x in zip(*knowledgess)] if len(knowledgess) > 0 else [[] for _ in batch['question']] # transpose the knowledge matrix
                # for i, (task, question, choices, answer_ix, knowledges) in enumerate(zip(batch['task'], batch['question'], batch['choices'], batch['answer_ix'], knowledgess)):
                #     item = {
                #         'task': task,
                #         'split': self.args.eval_split,
                #         'query': question,
                #         'cands': choices,
                #         'answer': choices[answer_ix],
                #         'knowledges': knowledges,
                #     }
                #     knowledge_outputs.append(copy.deepcopy(item))
                #     item.update({
                #         'scores_': results['answer_logitss'][:, i, :len(choices)].tolist(),
                #         'probs_': results['answer_probss'][:, i, :len(choices)].tolist(),
                #         'preds': choices[results['preds'][i].item()],
                #         'ok': int(results['corrects'][i]),
                #     })
                #     inference_outputs.append(item)

        corrects = torch.cat(corrects, dim=0) # (N)
        task_ixs = torch.cat(task_ixs, dim=0) # (N)

        corrects = self.accelerator.gather(corrects) # (num_gpus * N)
        task_ixs = self.accelerator.gather(task_ixs) # (num_gpus * N)

        # Accelerator may pad the tensors to make them divisible by the total batch size
        corrects = corrects[:len(self.eval_dataloader.dataset)]
        task_ixs = task_ixs[:len(self.eval_dataloader.dataset)]

        acc_weighted = corrects.float().mean().item()
        corrects_by_task = defaultdict(list)
        for task_ix, correct in zip(task_ixs, corrects):
            task = self.eval_dataloader.dataset.tasks[task_ix]
            corrects_by_task[task].append(correct)
        corrects_by_task = {k: torch.stack(v, dim=0) for k, v in corrects_by_task.items()}
        acc_by_task = {k: v.float().mean().item() for k, v in corrects_by_task.items()}
        acc_unweighted = np.mean(list(acc_by_task.values()))

        log.info(f'Evaluated [step {step}] acc_weighted = {acc_weighted:.4f} | acc_unweighted = {acc_unweighted:.4f}')
        log.info('Accuracy by task:')
        for task, acc in acc_by_task.items():
            log.info(f'\t{task} = {acc:.4f}')

        if not self.args.nosave and self.accelerator.is_main_process:
            stats = {
                'eval/step': step,
                'eval/acc_weighted': acc_weighted,
                'eval/acc_unweighted': acc_unweighted,
            }
            for task, acc in acc_by_task.items():
                stats[f'eval/acc/{task}'] = acc
            wandb.log(stats)

            knowledge_path = os.path.join(self.args.knowledge_dir, f'knowledge_rainier-ckp{step}.json')
            inference_path = os.path.join(self.args.inference_dir, f'inference_rainier-ckp{step}.knowledge_rainier-ckp{step}.json')
            with open(knowledge_path, 'w') as f:
                json.dump(knowledge_outputs, f, indent=4)
            with open(inference_path, 'w') as f:
                json.dump(inference_outputs, f, indent=4)

    """
    Internally set bias and gain terms based on the data from the dataloader
    """
    def set_reward_norm(self):
        accelerate.utils.wait_for_everyone()

        with torch.no_grad():
            rewards = []
            for i, batch in enumerate(tqdm(self.train_dataloader)):
                if self.args.eval_loop_cap is not None and i == self.args.eval_loop_cap:
                    break
                results = self.policy_model.sample(
                    questions_input_ids=batch['questions_input_ids'],
                    questions_attention_mask=batch['questions_attention_mask'],
                    temperature=self.args.temperature,
                )
                results = self.reward_model.get_reward(
                    questions_input_ids=batch['questions_input_ids'],
                    questions_attention_mask=batch['questions_attention_mask'],
                    choicess_input_ids=batch['choicess_input_ids'],
                    choicess_attention_mask=batch['choicess_attention_mask'],
                    choicess_labels=batch['choicess_labels'],
                    answer_ixs=batch['answer_ixs'],
                    knowledges_input_ids=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'knowledges_input_ids'],
                    knowledges_attention_mask=results[('' if self.args.do_not_lowercase else 'lowercased_') + 'knowledges_attention_mask'],
                    override_bias=0,
                    override_gain=1,
                )
                rewards.append(results['rewards/raw'])

        rewards = torch.cat(rewards, dim=0) # (N)
        rewards = self.accelerator.gather(rewards) # (num_gpus * N)
        rewards = rewards[:len(self.train_dataloader.dataset)] # remove padding

        old_mean, old_std = rewards.mean().item(), rewards.std().item()
        new_mean, new_std = 0.0, 1.0
        self.reward_model.gain = new_std / old_std
        self.reward_model.bias = new_mean - self.reward_model.gain * old_mean

        log.info(f'Reward normalization coefficients set to: gain = {self.reward_model.gain:.4f} | bias = {self.reward_model.bias:.4f}')

    def save(self, step):
        if self.args.nosave:
            return
        if step % self.args.save_interval != 0:
            return
        # this will overwrite an existing ckpt with the save filename!
        accelerate.utils.wait_for_everyone()
        if self.accelerator.distributed_type == accelerate.utils.DistributedType.FSDP:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.policy_model.model, StateDictType.FULL_STATE_DICT, save_policy):
                policy_model_state_dict = self.policy_model.model.state_dict()
            if not self.args.policy_value_sharing:
                with FSDP.state_dict_type(self.value_model.model, StateDictType.FULL_STATE_DICT, save_policy):
                    value_model_state_dict = self.value_model.model.state_dict()
        else:
            # TODO: test this!!
            policy_model_state_dict = self.accelerator.unwrap_model(self.policy_model.model).state_dict()
            if not self.args.policy_value_sharing:
                value_model_state_dict = self.accelerator.unwrap_model(self.value_model.model).state_dict()
        # TODO: Make optimizer state loading work
        # optimizer_state_dict = self.optimizer.state_dict()
        # scheduler_state_dict = self.scheduler.state_dict()
        result = {
            'model': policy_model_state_dict,
            # 'optimizer': optimizer_state_dict,
            # 'scheduler': scheduler_state_dict,
            'step': step,
            'eval_accs': self.eval_accs,
        }
        if not self.args.policy_value_sharing:
            result['value_model'] = value_model_state_dict
        self.accelerator.save(result, f'{self.args.model_dir}/last.pth')
        log.info(f'[step {step}] model checkpoint saved')


def main():
    args = get_args()

    set_seed(args.seed, args.cuda_deterministic)

    # GPUs
    accelerator = accelerate.Accelerator()
    device = accelerator.device

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
                args.save_dir = args.save_dir.replace('runs/', 'eval/')
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

        log.info(f'Write to output directory: {args.save_dir}')
        if accelerator.is_main_process:
            with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    # Load data
    log.info(f'Loading data ...')

    tokenizer = transformers.T5Tokenizer.from_pretrained(args.model_type)
    tokenizer.max_question_len = args.max_question_len
    tokenizer.max_answer_len = args.max_answer_len
    tokenizer.max_knowledge_len = args.max_knowledge_len

    if args.mode == 'train':
        train_dataset = QADataset(args, 'train', args.train_tasks, args.data_path, tokenizer)
        # train ds is shuffled in its constructor
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=train_dataset.collate_fn)
        log.info(f'Loaded train set with {len(train_dataset)} instances')

        eval_dataset = QADataset(args, 'dev', args.train_tasks, args.data_path, tokenizer)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=eval_dataset.collate_fn)
        log.info(f'Loaded dev set with {len(eval_dataset)} instances')

        train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)

    elif args.mode == 'eval':
        train_dataset = None
        train_dataloader = None

        eval_dataset = QADataset(args, args.eval_split, args.eval_tasks, args.data_path, tokenizer)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=eval_dataset.collate_fn)
        log.info(f'Loaded {args.eval_split} set with {len(eval_dataset)} instances')

        eval_dataloader = accelerator.prepare(eval_dataloader)

    # Initialize models and optimizer
    log.info(f'Initializing models ...')
    if args.mode == 'train':
        ref_policy = Policy(
            model_type=args.model_type,
            model_ckpt=args.model_ckpt,
            tokenizer=tokenizer,
            policy_value_sharing=args.policy_value_sharing,
            policy_reward_sharing=args.policy_reward_sharing,
            device=device,
        )
        ref_policy.model = accelerator.prepare(ref_policy.model)
        policy = Policy(
            model_type=args.model_type,
            model_ckpt=args.model_ckpt,
            tokenizer=tokenizer,
            policy_value_sharing=args.policy_value_sharing,
            policy_reward_sharing=args.policy_reward_sharing,
            device=device,
        )
        policy.model = accelerator.prepare(policy.model)
        value = Value(
            model_type=args.model_type,
            model_ckpt=args.model_ckpt if args.use_model_ckpt_for_value else None,
            model=policy.model if args.policy_value_sharing else None,
            tokenizer=tokenizer,
            device=device,
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
            device=device,
        )
        if not args.policy_reward_sharing:
            reward.model = accelerator.prepare(reward.model)

        # We never need to optimize the reward model's parameters separately!
        if args.policy_value_sharing:
            parameters = policy.model.parameters()
        else:
            parameters = chain(policy.model.parameters(), value.model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.lr, eps=1e-5)
        args.total_steps = ceil_div(args.total_episodes, args.batch_size * int(os.environ['SLURM_GPUS_ON_NODE']) * int(os.environ['SLURM_JOB_NUM_NODES']))
        warmup_steps = np.ceil(args.num_warmup_step_ratio * args.total_steps)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.total_steps)
        init_step = 0
        eval_accs = {}

        # Load from checkpoint if continue training
        # if args.load_from_ckpt is not None:
        #     checkpoint = torch.load(args.load_from_ckpt)
        #     policy.model.load_state_dict(checkpoint['model'], strict=False)
        #     if not args.policy_value_sharing:
        #         value.model.load_state_dict(checkpoint['value_model'], strict=False)
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     scheduler.load_state_dict(checkpoint['scheduler'])
        #     init_step = checkpoint['step']
        #     eval_accs = checkpoint['eval_accs']
        #     checkpoint.clear()

        #     # Reuse the reward normalization results
        #     reward.read_reward_norm(args.reward_dir)

        optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    elif args.mode == 'eval':
        ref_policy = None
        policy = Policy(
            model_type=args.model_type,
            model_ckpt=args.model_ckpt,
            tokenizer=tokenizer,
            policy_value_sharing=args.policy_value_sharing,
            policy_reward_sharing=args.policy_reward_sharing,
            device=device,
        )
        optimizer = None
        scheduler = None
        init_step = 0
        eval_accs = {}

        checkpoint = None
        if args.load_from_ckpt is not None:
            checkpoint = torch.load(args.load_from_ckpt, map_location='cpu')
            policy.model.load_state_dict(checkpoint['model'], strict=False) 
            init_step = checkpoint['step']
            checkpoint.clear()
        elif args.eval_ckpt is not None:
            checkpoint = torch.load(args.eval_ckpt, map_location='cpu')
            policy.model.load_state_dict(checkpoint, strict=False)
            checkpoint.clear()

        policy.model = accelerator.prepare(policy.model)

        value = None
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
            device=device,
        )

    # Set up trainer
    trainer = PPOTrainer(
        args=args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        ref_policy_model=ref_policy,
        policy_model=policy,
        value_model=value,
        reward_model=reward,
        optimizer=optimizer,
        scheduler=scheduler,
        init_step=init_step,
        eval_accs=eval_accs,
        device=device,
        accelerator=accelerator,
    )

    # Normalize the rewards to so that initially they have mean 0, var 1
    if args.mode == 'train':
        if args.load_from_ckpt is None:
            log.info('Setting reward norm')
            if args.gain is not None and args.bias is not None:
                reward.gain = args.gain
                reward.bias = args.bias
            else:
                trainer.set_reward_norm()
            log.info(f'Set reward norm as gain = {reward.gain}, bias = {reward.bias}')
            if not args.nosave and accelerator.is_main_process:
                reward.write_reward_norm(args.reward_dir)

    # Evaluate baseline (no knowledge)
    if args.eval_baseline:
        trainer.eval(step=-1)

    # Train or evaluate
    if args.mode == 'train':
        pbar = tqdm(list(range(init_step, args.total_steps + 1)))
        for step in pbar:
            trainer.train(step)
    elif args.mode == 'eval':
        trainer.eval(init_step)


if __name__ == '__main__':
    main()
