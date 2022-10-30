import os
import json
import copy
from collections import defaultdict
import argparse
import itertools
from tqdm import tqdm
import logging
from typing import Dict
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from utils.utils import whiten, reduce_mean, reduce_sum, clamp
from model.policy import Policy
from model.value import Value
from model.reward import Reward


class PPOTrainer:

    def __init__(self,
                 args: argparse.Namespace,
                 train_dataloader: DataLoader,
                 eval_dataloader: DataLoader,
                 policy_model: Policy,
                 ref_policy_model: Policy,
                 value_model: Value,
                 reward_model: Reward,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LambdaLR,
                 init_step: int,
                 eval_accs: Dict,
                 log: logging.Logger,
                ):
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.train_sampler = iter(self.train_dataloader) if self.train_dataloader is not None else None
        self.policy_model = policy_model
        self.ref_policy_model = ref_policy_model
        self.value_model = value_model
        self.reward_model = reward_model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.log = log
        if not args.nosave:
            # self.writer = SummaryWriter(log_dir=args.tensorboard_dir)
            wandb.init(project='rainier_stageII' if args.mode == 'train' else 'rainier_eval', name=args.run_name, config=args)
            wandb.define_metric('train/step')
            wandb.define_metric('eval/step')
            wandb.define_metric('train/*', step_metric='train/step')
            wandb.define_metric('eval/*', step_metric='eval/step', summary='max')

        if self.train_dataloader is not None:
            self.train_sampler = iter(self.train_dataloader)
            for _ in range(init_step % len(self.train_dataloader)):
                next(self.train_sampler)

        self.eval_accs = eval_accs

    def loss(self, results):
        old_values = results['response/value']
        old_logprobs = results['response/logprobs']
        rewards = results['rewards/penalized']
        mask = results['response/mask']

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

        forward_inputs = {'query_input_ids': results['query/input_ids'],
                          'query_mask': results['query/mask'],
                          'response_input_ids': results['response/input_ids'],
                          'response_mask': results['response/mask']}

        policy_forward = self.policy_model.forward_pass(**forward_inputs)
        new_logprobs = policy_forward['response/logprobs']

        ratio = torch.exp(new_logprobs - old_logprobs)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, min=1.0 - self.args.cliprange, max=1.0 + self.args.cliprange)
        pg_loss = reduce_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = reduce_mean(torch.gt(pg_losses2, pg_losses).float(), mask)

        value_forward = self.value_model.forward_pass(**forward_inputs)
        new_values = value_forward['response/value']
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
        self.valid(step)
        self.save(step)

        try:
            batch = next(self.train_sampler)
        except StopIteration:
            self.train_sampler = iter(self.train_dataloader)
            batch = next(self.train_sampler)

        # Rollout from current policy
        with torch.no_grad():
            results = self.policy_model.sample(
                text=batch['question'],
                temperature=self.args.temperature,
            )
        # if not self.args.nosave:
        #     self.writer.add_text('Question-Knowledge', f'{results["query/text"][0]} ==> {results["response/text"][0]}', global_step=step)

        forward_inputs = {'query_input_ids': results['query/input_ids'],
                          'query_mask': results['query/mask'],
                          'response_input_ids': results['response/input_ids'],
                          'response_mask': results['response/mask']}

        # Run value network
        with torch.no_grad(): # treat the values at beginning of step as ground-truth
            value_forward = self.value_model.forward_pass(**forward_inputs)
            results['response/value'] = value_forward['response/value'].to(self.policy_model.device)
            results['response/value'] *= results['response/mask']  # TODO: I doubt if this line is necessary

        # Run ref policy
        with torch.no_grad():
            ref_policy_forward = self.ref_policy_model.forward_pass(**forward_inputs)
            results['response/ref_logits'] = ref_policy_forward['response/logits'].to(self.policy_model.device)
            results['response/ref_logprobs'] = ref_policy_forward['response/logprobs'].to(self.policy_model.device)

        # Get reward
        with torch.no_grad():
            reward_results = self.reward_model.get_reward(
                questions=batch['question'],
                knowledges=results['response/text'],
                choicess=batch['choices'],
                answer_ixs=batch['answer_ix'],
            )
            results = {**results, **reward_results}
            self.reward_model.kl_penalize_reward(results)

        # Train
        # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
        for ppo_epoch_idx in range(self.args.noptepochs):
            self.optimizer.zero_grad()
            self.loss(results)
            results['loss/total'].backward()
            if self.args.clip_grad:
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(self.policy_model.model.parameters(),
                                    self.value_model.model.parameters()),
                    self.args.max_grad_norm)
            self.optimizer.step()

        # Increment scheduler
        self.scheduler.step()

        # Logging
        if not self.args.nosave and step % self.args.log_interval == 0:
            stats = {
                'train/step': step,
                'train/acc': np.mean(results['corrects']),
                'train/loss/total': results['loss/total'].item(),
                'train/loss/policy': results['loss/policy'].item(),
                'train/loss/value': results['loss/value'].item(),
                'train/reward/penalized': torch.mean(reduce_sum(results['rewards/penalized'], results['response/mask'], axis=1)).item(),
                'train/reward/KL': torch.mean(reduce_sum(results['rewards/kl'], results['response/mask'], axis=1)).item(),
                'train/reward/normalized': np.mean(results['rewards/normalized']),
                'train/reward/raw': np.mean(results['rewards/raw']),
            }
            wandb.log(stats)
            # for k, v in stats.items():
            #     self.writer.add_scalar(k, v, step)

    def valid(self, step):
        if step % self.args.eval_interval != 0:
            return
        if step in self.eval_accs:
            return
        self.log.info(f'Evaluating [ppo_step {step}] ...')

        corrects = []
        corrects_by_task = defaultdict(list)
        results_table = wandb.Table(columns=['step', 'id', 'task', 'question', 'knowledge', 'pred', 'answer_ix', 'correct'])

        for i, batch in enumerate(tqdm(self.eval_dataloader)):
            with torch.no_grad():
                rollouts = self.policy_model.sample(
                    text=batch['question'],
                    top_p=0.0,
                )
                knowledges = rollouts['response/text']

                results = self.reward_model.get_reward(
                    questions=batch['question'],
                    knowledges=knowledges,
                    choicess=batch['choices'],
                    answer_ixs=batch['answer_ix'], 
                    override_bias=0,
                    override_gain=1,
                )

            corrects += results['corrects']
            for task, c in zip(batch['task'], results['corrects']):
                corrects_by_task[task].append(c)

            results_table.add_data(step, i, batch['task'][0], batch['question'][0], knowledges[0], results['preds'][0], batch['answer_ix'][0], results['corrects'][0])

        acc_weighted = np.mean(corrects)
        acc_by_task = {k: np.mean(v) for k, v in corrects_by_task.items()}
        acc_unweighted = np.mean(list(acc_by_task.values()))

        self.log.info(f'Evaluated [ppo_step {step}] acc_weighted = {acc_weighted:.4f} | acc_unweighted = {acc_unweighted:.4f}')
        self.log.info('Accuracy by task:')
        for task, acc in acc_by_task.items():
            self.log.info(f'\t{task} = {acc:.4f}')

        if self.args.nosave:
            self.eval_accs[step] = acc_unweighted
        else:
            # self.writer.add_scalar('eval/acc_weighted', acc_weighted, step)
            # self.writer.add_scalar('eval/acc_unweighted', acc_unweighted, step)
            # for task, acc in acc_by_task.items():
            #     self.writer.add_scalar(f'eval/acc/{task}', acc, step)
            stats = {
                'eval/step': step,
                'eval/results_table': results_table,
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
                        self.log.warning(f'Cannot remove previous best ckpt!')
                self.save(step, last=False)
                self.log.info(f'Best ckpt updated to [step {step}]')

    def eval(self, step): # step=-1 for baseline
        self.log.info(f'Evaluating [ppo_step {step}] ...')

        corrects = []
        corrects_by_task = defaultdict(list)
        knowledge_outputs = []
        inference_outputs = []

        for i, batch in enumerate(tqdm(self.eval_dataloader)):
            with torch.no_grad():
                knowledgess = []
                # If not baseline, generate knowledge
                if step != -1:
                    for j in range(self.args.num_samples):
                        rollouts = self.policy_model.sample(
                            text=batch['question'],
                            top_p=self.args.top_p,
                        )
                        knowledgess.append(rollouts['response/text'])

                results = self.reward_model.get_reward_ensemble(
                    questions=batch['question'],
                    knowledgess=knowledgess,
                    choicess=batch['choices'],
                    answer_ixs=batch['answer_ix'], 
                    override_bias=0,
                    override_gain=1,
                )

            corrects += results['corrects']
            for task, c in zip(batch['task'], results['corrects']):
                corrects_by_task[task].append(c)

            knowledgess = [list(x) for x in zip(*knowledgess)] if len(knowledgess) > 0 else [[] for _ in batch['question']] # transpose the knowledege matrix
            for i, (task, question, choices, answer_ix, knowledges) in enumerate(zip(batch['task'], batch['question'], batch['choices'], batch['answer_ix'], knowledgess)):
                item = {
                    'task': task,
                    'split': self.args.eval_split,
                    'query': question,
                    'cands': choices,
                    'answer': choices[answer_ix],
                    'knowledges': knowledges,
                }
                knowledge_outputs.append(copy.deepcopy(item))
                item.update({
                    'scores_': results['answer_logitss'][:, i, :len(choices)].tolist(),
                    'probs_': results['answer_probss'][:, i, :len(choices)].tolist(),
                    'preds': choices[results['preds'][i].item()],
                    'ok': int(results['corrects'][i]),
                })
                inference_outputs.append(item)

        acc_weighted = np.mean(corrects)
        acc_by_task = {k: np.mean(v) for k, v in corrects_by_task.items()}
        acc_unweighted = np.mean(list(acc_by_task.values()))

        self.log.info(f'Evaluated [ppo_step {step}] acc_weighted = {acc_weighted:.4f} | acc_unweighted = {acc_unweighted:.4f}')
        self.log.info('Accuracy by task:')
        for task, acc in acc_by_task.items():
            self.log.info(f'\t{task} = {acc:.4f}')

        # self.writer.add_scalar('eval/acc_weighted', acc_weighted, step)
        # self.writer.add_scalar('eval/acc_unweighted', acc_unweighted, step)
        # for task, acc in acc_by_task.items():
        #     self.writer.add_scalar(f'eval/acc/{task}', acc, step)
        stats = {
            'eval/step': step,
            'eval/acc_weighted': acc_weighted,
            'eval/acc_unweighted': acc_unweighted,
        }
        for task, acc in acc_by_task.items():
            stats[f'eval/acc/{task}'] = acc
        wandb.log(stats)

        knowledge_path = os.path.join(self.args.knowledge_dir, f'knowledge_rainier-ckp{step}.json')
        inference_path = os.path.join(self.args.inference_dir, f'inference_{self.args.qa_model_type.split("/")[-1]}.knowledge_rainier-ckp{step}.json')
        with open(knowledge_path, 'w') as f:
            json.dump(knowledge_outputs, f, indent=4)
        with open(inference_path, 'w') as f:
            json.dump(inference_outputs, f, indent=4)

    """
    Internally set bias and gain terms based on the data from the dataloader
    """
    def set_reward_norm(self):
        rewards = []
        for batch in tqdm(self.train_dataloader):
            results = self.policy_model.sample(
                text=batch['question'],
                temperature=self.args.temperature,
            )
            reward_results = self.reward_model.get_reward(
                questions=batch['question'],
                knowledges=results['response/text'],
                choicess=batch['choices'],
                answer_ixs=batch['answer_ix'],
                override_bias=0,
                override_gain=1,
            )
            rewards += reward_results['rewards/raw']
        old_mean, old_std = np.mean(rewards), np.std(rewards)
        new_mean, new_std = 0.0, 1.0
        self.reward_model.gain = new_std / old_std
        self.reward_model.bias = new_mean - self.reward_model.gain * old_mean

    def save(self, step, last=True):
        if self.args.nosave:
            return
        if step % self.args.save_interval != 0:
            return
        # this will overwrite an existing ckpt with the save filename!
        torch.save({
            'policy_model': self.policy_model.model.state_dict(),
            'value_model': self.value_model.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'step': step,
            'eval_accs': self.eval_accs,
        }, f'{self.args.model_dir}/{"last" if last else "ckp_" + str(step)}.pth')
        self.log.info(f'[ppo_step {step}] model checkpoint saved')

