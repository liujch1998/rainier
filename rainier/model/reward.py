import json
import os
from typing import Optional, List, Iterable, Dict, Any, Tuple
from itertools import chain
import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils.utils import reduce_mean


class Reward:

    def __init__(self,
                 model_type,
                 model_ckpt,
                 max_input_len,
                 batch_size,
                 reward_shape,
                 kl_coef,
                 ensembling,
                 device: torch.device,
                ):
        self.tokenizer = T5Tokenizer.from_pretrained(model_type)
        self.inference_model = T5ForConditionalGeneration.from_pretrained(model_ckpt if model_ckpt is not None else model_type)
        self.inference_model.eval()
        self.inference_model.to(device)

        self.gain, self.bias = None, None
        self.max_input_len = max_input_len
        self.batch_size = batch_size
        self.reward_shape = reward_shape
        self.kl_coef = kl_coef
        self.ensembling = ensembling

        self.device = device

        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100,reduction='none')

    """
    questions: list of strings
    knowledges: list of knowledges, 1 to 1 mapping to questions
    choicess: list of lists of candidate choices for each question
    answer_ixs: list of integer indeces corresponding to ground truth answer index from answers list of lists
    """
    def get_reward(self,
                   questions: List[str],
                   knowledges: List[str],
                   choicess: List[List[str]],
                   answer_ixs: List[int],
                   override_gain = None,
                   override_bias = None,
                   skip_reward = False,
                  ) -> Tuple[List[float], float, int, int]:
        if knowledges is None:
            knowledges = [None for _ in questions]

        assert len(questions) == len(knowledges)
        assert len(questions) == len(choicess)

        questions = [a.lower() for a in questions]
        knowledges = [a.lower() if a is not None else None for a in knowledges]
        choicess = [[a.lower() for a in b] for b in choicess]
        prompts = [
            question + (f' \\n {knowledge}' if knowledge is not None else '')
            for question, knowledge, choices in zip(questions, knowledges, choicess)]

        # Compute number of choices for each question, and flatten prompts accordingly
        num_ans = [len(c) for c in choicess]
        max_ans_num = max(num_ans)
        flattened_prompts = list(np.repeat(np.array(prompts, dtype=object), num_ans, axis=0))
        flattened_choices = list(chain(*choicess))

        # Preallocate tensor for all of the loss
        all_losses = torch.zeros(len(flattened_prompts), device=self.device)

        for i in range(0, len(flattened_prompts), self.batch_size):
            j = min(i + self.batch_size, len(flattened_prompts))
            batch_prompts = flattened_prompts[i:j]
            batch_choices = flattened_choices[i:j]

            # Tokenize prompts and inputs
            tokenized_prompts = self.tokenizer(batch_prompts, return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.max_input_len).to(self.device)
            tokenized_choices = self.tokenizer(batch_choices, return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.max_input_len).to(self.device)
            tokenized_choices_ids = tokenized_choices.input_ids
            pad_mask = (tokenized_choices_ids == self.tokenizer.pad_token_id)

            with torch.no_grad():
                logits = self.inference_model(
                    input_ids=tokenized_prompts.input_ids,
                    attention_mask=tokenized_prompts.attention_mask,
                    labels=tokenized_choices_ids,
                ).logits # (B, L, V)

            # Set ignore index for loss calculation
            tokenized_choices_ids[pad_mask] = -100

            # Loss will be exactly 0 for ignored, pad idx tokens
            losses = self.loss_fct(logits.view(-1, logits.size(-1)), tokenized_choices_ids.view(-1))

            # Take mean of loss
            losses = losses.view(tokenized_choices_ids.shape) # (B, L)
            losses = reduce_mean(losses, ~pad_mask, axis=-1) # (B)

            # Update all loss
            all_losses[i:j] = losses

        # Now, convert back to tensor of the correct shape - # of questions X max # of answers
        answer_logits = torch.empty(len(questions), max_ans_num, device=self.device).fill_(float('-inf'))
        cur_arr_idx = 0
        for idx, sz in enumerate(num_ans):
            answer_logits[idx, :sz] = -all_losses[cur_arr_idx:cur_arr_idx+sz]
            cur_arr_idx += sz
        answer_probs = answer_logits.softmax(axis=1)

        # Compute accuracy from argmax answer
        preds = answer_probs.argmax(axis=1).detach().cpu()
        corrects = (preds == torch.tensor(answer_ixs)).tolist()

        if skip_reward:
            return {
                'corrects': corrects,
                'preds': preds,
                'answer_logits': answer_logits.detach().cpu(),
                'answer_probs': answer_probs.detach().cpu(),
            }

        # Probabilities of the gt answer
        gt_idxs = torch.tensor(answer_ixs, dtype=torch.long, device=self.device).unsqueeze(-1)
        gt_answer_probs = torch.gather(answer_probs, dim=-1, index=gt_idxs)
        gt_answer_logits = torch.gather(answer_logits, dim=-1, index=gt_idxs)

        if self.reward_shape == 0: # r = p(a*|q,k) - p(a*|q), same as using --prob_diff before
            rewards = gt_answer_probs.squeeze(-1)

            if knowledges[0] is not None:
                knowless_results = self.get_reward(
                    questions, None, choicess, answer_ixs, override_gain, override_bias)
                rewards -= torch.tensor(knowless_results['rewards/raw'], device=self.device)

        elif self.reward_shape == 1: # r = p(a*|q,k), same as using no --prob_diff before
            rewards = gt_answer_probs.squeeze(-1)

        elif self.reward_shape == 2: # r = s(a*|q,k) - s(a*|q), same as using --prob_diff and --prob_nonorm before
            rewards = gt_answer_logits.squeeze(-1)

            if knowledges[0] is not None:
                knowless_results = self.get_reward(
                    questions, None, choicess, answer_ixs, override_gain, override_bias)
                rewards -= torch.tensor(knowless_results['rewards/raw'], device=self.device)

        elif self.reward_shape == 3: # r = s(a*|q,k), same as using --prob_nonorm but no --prob_diff before
            rewards = gt_answer_logits.squeeze(-1)

        elif self.reward_shape == 4: # r = { tanh[s(a*|q,k) - max s(a'|q,k)] - tanh[s(a*|q) - max s(a'|q)] } / 2
            distractor_idxs = answer_logits.argsort(dim=1, descending=True)
            take_second = (distractor_idxs[:, :1] == gt_idxs).long()

            distractor_idxs = torch.gather(distractor_idxs, dim=-1, index=take_second)
            distractor_logits = torch.gather(answer_logits, dim=-1, index=distractor_idxs)
            rewards = (0.5 * torch.tanh(gt_answer_logits - distractor_logits)).squeeze(-1)

            if knowledges[0] is not None:
                knowless_results = self.get_reward(
                    questions, None, choicess, answer_ixs, override_gain, override_bias)
                rewards -= torch.tensor(knowless_results['rewards/raw'], device=self.device)

        elif self.reward_shape == 5: # r = { sgn[s(a*|q,k) - max s(a'|q,k)] - sgn[s(a*|q) - max s(a'|q)] } / 2
            distractor_idxs = answer_logits.argsort(dim=1, descending=True)
            take_second = (distractor_idxs[:, :1] == gt_idxs).long()

            distractor_idxs = torch.gather(distractor_idxs, dim=-1, index=take_second)
            distractor_logits = torch.gather(answer_logits, dim=-1, index=distractor_idxs)
            rewards = (0.5 * torch.sign(gt_answer_logits - distractor_logits)).squeeze(-1)

            if knowledges[0] is not None:
                knowless_results = self.get_reward(
                    questions, None, choicess, answer_ixs, override_gain, override_bias)
                rewards -= torch.tensor(knowless_results['rewards/raw'], device=self.device)

        elif self.reward_shape == 6: # r = { tanh[p(a*|q,k) - 1/|A|] - tanh[p(a*|q) - 1/|A|] } / 2
            rewards = (0.5 * torch.tanh(gt_answer_probs - 1.0 / torch.tensor(num_ans, device=gt_answer_probs.device).unsqueeze(-1))).squeeze(-1)

            if knowledges[0] is not None:
                knowless_results = self.get_reward(
                    questions, None, choicess, answer_ixs, override_gain, override_bias)
                rewards -= torch.tensor(knowless_results['rewards/raw'], device=self.device)

        elif self.reward_shape == 7: # r = { sign[p(a*|q,k) - 1/|A|] - sign[p(a*|q) - 1/|A|] } / 2
            rewards = (0.5 * torch.sign(gt_answer_probs - 1.0 / torch.tensor(num_ans, device=gt_answer_probs.device).unsqueeze(-1))).squeeze(-1)

            if knowledges[0] is not None:
                knowless_results = self.get_reward(
                    questions, None, choicess, answer_ixs, override_gain, override_bias)
                rewards -= torch.tensor(knowless_results['rewards/raw'], device=self.device)

        rewards_raw = rewards.detach().cpu().tolist()

        gain = self.gain if override_gain is None else override_gain
        bias = self.bias if override_bias is None else override_bias
        rewards_normalized = [gain * x + bias for x in rewards_raw]

        return {
            'corrects': corrects,
            'preds': preds,
            'answer_logits': answer_logits.detach().cpu(),
            'answer_probs': answer_probs.detach().cpu(),
            'rewards/raw': rewards_raw,
            'rewards/normalized': rewards_normalized,
        }

    def kl_penalize_reward(self, results):
        logprobs = results['response/logprobs']
        ref_logprobs = results['response/ref_logprobs']
        mask = results['response/mask']
        normalized_rewards = results['rewards/normalized']

        kl = logprobs - ref_logprobs
        kl_penalty = self.kl_coef * kl
        RL = logprobs.size(1)
        flattened_rewards = torch.tensor([
            [0.] * (l-1) + [r] + [0.] * (RL-l)
            for r, l in zip(normalized_rewards, torch.sum(mask, dim=1).tolist())
        ], device=logprobs.device) # (B, RL)
        penalized_rewards = flattened_rewards - kl_penalty
        # TODO: This is slightly different from the paper

        results['rewards/kl'] = kl
        results['rewards/kl_penalty'] = kl_penalty
        results['rewards/penalized'] = penalized_rewards

    def get_reward_ensemble(self,
                            questions: List[str],
                            knowledgess: List[List[str]],
                            choicess: List[List[str]],
                            answer_ixs: List[int],
                            override_gain = None,
                            override_bias = None,
                           ) -> Tuple[List[float], float, int, int]:
        answer_logitss = []
        answer_probss = []

        knowless_results = self.get_reward(questions, None, choicess, answer_ixs, override_gain, override_bias)
        answer_logitss.append(knowless_results['answer_logits'])
        answer_probss.append(knowless_results['answer_probs'])

        for knowledges in knowledgess:
            results = self.get_reward(questions, knowledges, choicess, answer_ixs, override_gain, override_bias, skip_reward=True)
            answer_logitss.append(results['answer_logits'])
            answer_probss.append(results['answer_probs'])

        answer_logitss = torch.stack(answer_logitss) # (K+1, B, C)
        answer_probss = torch.stack(answer_probss)

        if self.ensembling == 'max':
            answer_probs = answer_probss.max(dim=0).values
            preds = answer_probs.argmax(dim=1)
        elif self.ensembling == 'moe':
            answer_probs = answer_probss.mean(dim=0)
            preds = answer_probs.argmax(dim=1)
        elif self.ensembling == 'poe':
            answer_probs = answer_probss.log().mean(dim=0).exp()
            preds = answer_probs.argmax(dim=1)
        elif self.ensembling == 'majority':
            predss = answer_probss.argmax(dim=2) # (K+1, B)
            preds = predss.mode(dim=0).values

        num_ans = [len(c) for c in choicess]
        max_ans_num = max(num_ans)

        # Compute accuracy from argmax answer
        corrects = (preds == torch.tensor(answer_ixs)).tolist()

        '''
        selected_knowledges_ix = (answer_probss.max(dim=-1).values.argmax(dim=0) - 1).tolist() # (B)
        selected_knowledges = [knowledgess[ix][b] if ix >= 0 else '' for (b, ix) in enumerate(selected_knowledges_ix)]
        knowless_corrects = (knowless_results['answer_probs'].argmax(dim=1) == torch.Tensor(answer_ixs)).tolist()
        knowful_corrects = (pred_answers == torch.Tensor(answer_ixs)).tolist()
        knowledge_analyses = list(zip(selected_knowledges, knowless_corrects, knowful_corrects))
        '''

        return {
            'corrects': corrects,
            'preds': preds,
            'answer_logitss': answer_logitss,
            'answer_probss': answer_probss,
        }

    def write_reward_norm(self, reward_dir):
        reward_dict = {
            'gain': self.gain,
            'bias': self.bias,
        }
        with open(os.path.join(reward_dir, 'reward_normalization.json'), 'w') as f:
            json.dump(reward_dict, f, indent=4)

    def read_reward_norm(self, reward_dir):
        with open(os.path.join(reward_dir, 'reward_normalization.json')) as f:
            reward_dict = json.load(f)
        self.gain = reward_dict['gain']
        self.bias = reward_dict['bias']

