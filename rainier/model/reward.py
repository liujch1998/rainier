import json
import os
from typing import Optional, List, Iterable, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from utils.utils import reduce_mean


class Reward:

    def __init__(self,
                 model_type,
                 model_ckpt,
                 model,
                 tokenizer,
                 batch_size,
                 reward_shape,
                 kl_coef,
                 ensembling,
                 do_not_lowercase,
                 no_knowless_expert,
                ):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.reward_shape = reward_shape
        self.kl_coef = kl_coef
        self.ensembling = ensembling
        self.do_not_lowercase = do_not_lowercase
        self.no_knowless_expert = no_knowless_expert

        self.gain, self.bias = None, None

        if model is not None:
            self.model = model
            return

        self.model = T5ForConditionalGeneration.from_pretrained(model_ckpt if model_ckpt is not None else model_type)
        self.model.eval()

    """
    questions: list of strings
    knowledges: list of knowledges, 1 to 1 mapping to questions
    choicess: list of lists of candidate choices for each question
    answer_ixs: list of integer indeces corresponding to ground truth answer index from answers list of lists
    """
    def get_reward(self,
                   questions_input_ids: torch.tensor, # (B, QL)
                   questions_attention_mask: torch.tensor, # (B, QL)
                   choicess_input_ids: torch.tensor, # (B, C, AL)
                   choicess_attention_mask: torch.tensor, # (B, C, AL)
                   choicess_labels: torch.tensor, # (B, C, AL)
                   answers_labels: torch.tensor, # (B, AL)
                   answer_ixs: torch.tensor, # (B)
                   knowledges_input_ids: Optional[torch.tensor] = None, # (B, KL)
                   knowledges_attention_mask: Optional[torch.tensor] = None, # (B, KL)
                   override_gain = None,
                   override_bias = None,
                   skip_reward = False,
                  ):
        questions_len = questions_attention_mask.sum(dim=1)
        if knowledges_input_ids is not None and knowledges_attention_mask is not None:
            prompts_input_ids = F.pad(questions_input_ids, (0, self.tokenizer.max_knowledge_len + 2), value=self.tokenizer.pad_token_id)
            prompts_attention_mask = F.pad(questions_attention_mask, (0, self.tokenizer.max_knowledge_len + 2), value=0)
            B = questions_input_ids.size(0)
            knowledges_len = knowledges_attention_mask.sum(dim=1)
            for b in range(B):
                if knowledges_len[b] == 1: # Empty knowledge
                    continue
                prompts_input_ids[b, questions_len[b]-1:questions_len[b]+2] = torch.tensor([3, 2, 29], dtype=torch.long, device=questions_input_ids.device)
                prompts_input_ids[b, questions_len[b]+2:questions_len[b]+2+knowledges_len[b]] = knowledges_input_ids[b, :knowledges_len[b]]
                prompts_attention_mask[b, questions_len[b]-1:questions_len[b]+2+knowledges_len[b]] = 1
        else:
            prompts_input_ids = questions_input_ids
            prompts_attention_mask = questions_attention_mask
        prompts_text = self.tokenizer.batch_decode(prompts_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # Compute number of choices for each question, and flatten prompts accordingly
        num_ans = (choicess_labels[:, :, 0] != 1).long().sum(dim=1) # (B)
        flattened_prompts_input_ids = torch.repeat_interleave(prompts_input_ids, choicess_input_ids.size(1), dim=0) # (B * C, QL + KL)
        flattened_prompts_attention_mask = torch.repeat_interleave(prompts_attention_mask, choicess_input_ids.size(1), dim=0) # (B * C, QL + KL)
        flattened_choices_input_ids = choicess_input_ids.flatten(0, 1) # (B * C, AL)
        flattened_choices_attention_mask = choicess_attention_mask.flatten(0, 1) # (B * C, AL)
        flattened_choices_labels = choicess_labels.flatten(0, 1) # (B * C, AL)

        all_losses = []
        assert flattened_choices_input_ids.size(0) % self.batch_size == 0

        for i in range(0, flattened_prompts_input_ids.size(0), self.batch_size):
            j = min(i + self.batch_size, flattened_prompts_input_ids.size(0))
            batch_prompts_input_ids = flattened_prompts_input_ids[i:j]
            batch_prompts_attention_mask = flattened_prompts_attention_mask[i:j]
            batch_choices_input_ids = flattened_choices_input_ids[i:j]
            batch_choices_attention_mask = flattened_choices_attention_mask[i:j]
            batch_choices_labels = flattened_choices_labels[i:j]

            with torch.no_grad():
                logits = self.model(
                    input_ids=batch_prompts_input_ids,
                    attention_mask=batch_prompts_attention_mask,
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

        # Compute QA loss
        qa_loss = self.model(
            input_ids=prompts_input_ids,
            attention_mask=prompts_attention_mask,
            labels=answers_labels,
        ).loss

        if skip_reward:
            return {
                'corrects': corrects,
                'preds': preds,
                'answer_logitss': answer_logitss, # (B, C)
                'answer_probss': answer_probss, # (B, C)
                'prompts_text': prompts_text,
                'prompts_input_ids': prompts_input_ids,
                'prompts_attention_mask': prompts_attention_mask,
                'loss/qa' if knowledges_input_ids is None else 'loss/qka': qa_loss,
            }

        # Probabilities of the gt answer
        gt_idxs = answer_ixs.unsqueeze(-1)
        gt_answer_probs = torch.gather(answer_probss, dim=-1, index=gt_idxs)
        gt_answer_logits = torch.gather(answer_logitss, dim=-1, index=gt_idxs)

        if self.reward_shape == 0: # r = p(a*|q,k) - p(a*|q), same as using --prob_diff before
            rewards = gt_answer_probs.squeeze(-1)

            if knowledges_input_ids is not None:
                knowless_results = self.get_reward(
                    questions_input_ids,
                    questions_attention_mask,
                    choicess_input_ids,
                    choicess_attention_mask,
                    choicess_labels,
                    answers_labels,
                    answer_ixs,
                    knowledges_input_ids=None,
                    knowledges_attention_mask=None,
                    override_gain=override_gain,
                    override_bias=override_bias,
                )
                rewards -= knowless_results['rewards/raw']
                knowless_qa_loss = knowless_results['loss/qa']

        elif self.reward_shape == 1: # r = p(a*|q,k), same as using no --prob_diff before
            rewards = gt_answer_probs.squeeze(-1)
            if knowledges_input_ids is not None:
                knowless_results = self.get_reward(
                    questions_input_ids,
                    questions_attention_mask,
                    choicess_input_ids,
                    choicess_attention_mask,
                    choicess_labels,
                    answers_labels,
                    answer_ixs,
                    knowledges_input_ids=None,
                    knowledges_attention_mask=None,
                    override_gain=override_gain,
                    override_bias=override_bias,
                )
                knowless_qa_loss = knowless_results['loss/qa']

        elif self.reward_shape == 2: # r = s(a*|q,k) - s(a*|q), same as using --prob_diff and --prob_nonorm before
            rewards = gt_answer_logits.squeeze(-1)

            if knowledges_input_ids is not None:
                knowless_results = self.get_reward(
                    questions_input_ids,
                    questions_attention_mask,
                    choicess_input_ids,
                    choicess_attention_mask,
                    choicess_labels,
                    answers_labels,
                    answer_ixs,
                    knowledges_input_ids=None,
                    knowledges_attention_mask=None,
                    override_gain=override_gain,
                    override_bias=override_bias,
                )
                rewards -= knowless_results['rewards/raw']
                knowless_qa_loss = knowless_results['loss/qa']

        elif self.reward_shape == 3: # r = s(a*|q,k), same as using --prob_nonorm but no --prob_diff before
            rewards = gt_answer_logits.squeeze(-1)
            if knowledges_input_ids is not None:
                knowless_results = self.get_reward(
                    questions_input_ids,
                    questions_attention_mask,
                    choicess_input_ids,
                    choicess_attention_mask,
                    choicess_labels,
                    answers_labels,
                    answer_ixs,
                    knowledges_input_ids=None,
                    knowledges_attention_mask=None,
                    override_gain=override_gain,
                    override_bias=override_bias,
                )
                knowless_qa_loss = knowless_results['loss/qa']

        elif self.reward_shape == 4: # r = { tanh[s(a*|q,k) - max s(a'|q,k)] - tanh[s(a*|q) - max s(a'|q)] } / 2
            distractor_idxs = answer_logitss.argsort(dim=1, descending=True)
            take_second = (distractor_idxs[:, :1] == gt_idxs).long()

            distractor_idxs = torch.gather(distractor_idxs, dim=-1, index=take_second)
            distractor_logits = torch.gather(answer_logitss, dim=-1, index=distractor_idxs)
            rewards = (0.5 * torch.tanh(gt_answer_logits - distractor_logits)).squeeze(-1)

            if knowledges_input_ids is not None:
                knowless_results = self.get_reward(
                    questions_input_ids,
                    questions_attention_mask,
                    choicess_input_ids,
                    choicess_attention_mask,
                    choicess_labels,
                    answers_labels,
                    answer_ixs,
                    knowledges_input_ids=None,
                    knowledges_attention_mask=None,
                    override_gain=override_gain,
                    override_bias=override_bias,
                )
                rewards -= knowless_results['rewards/raw']
                knowless_qa_loss = knowless_results['loss/qa']

        elif self.reward_shape == 5: # r = { sgn[s(a*|q,k) - max s(a'|q,k)] - sgn[s(a*|q) - max s(a'|q)] } / 2
            distractor_idxs = answer_logitss.argsort(dim=1, descending=True)
            take_second = (distractor_idxs[:, :1] == gt_idxs).long()

            distractor_idxs = torch.gather(distractor_idxs, dim=-1, index=take_second)
            distractor_logits = torch.gather(answer_logitss, dim=-1, index=distractor_idxs)
            rewards = (0.5 * torch.sign(gt_answer_logits - distractor_logits)).squeeze(-1)

            if knowledges_input_ids is not None:
                knowless_results = self.get_reward(
                    questions_input_ids,
                    questions_attention_mask,
                    choicess_input_ids,
                    choicess_attention_mask,
                    choicess_labels,
                    answers_labels,
                    answer_ixs,
                    knowledges_input_ids=None,
                    knowledges_attention_mask=None,
                    override_gain=override_gain,
                    override_bias=override_bias,
                )
                rewards -= knowless_results['rewards/raw']
                knowless_qa_loss = knowless_results['loss/qa']

        elif self.reward_shape == 6: # r = { tanh[p(a*|q,k) - 1/|A|] - tanh[p(a*|q) - 1/|A|] } / 2
            rewards = (0.5 * torch.tanh(gt_answer_probs - 1.0 / torch.tensor(num_ans, device=gt_answer_probs.device).unsqueeze(-1))).squeeze(-1)

            if knowledges_input_ids is not None:
                knowless_results = self.get_reward(
                    questions_input_ids,
                    questions_attention_mask,
                    choicess_input_ids,
                    choicess_attention_mask,
                    choicess_labels,
                    answers_labels,
                    answer_ixs,
                    knowledges_input_ids=None,
                    knowledges_attention_mask=None,
                    override_gain=override_gain,
                    override_bias=override_bias,
                )
                rewards -= knowless_results['rewards/raw']
                knowless_qa_loss = knowless_results['loss/qa']

        elif self.reward_shape == 7: # r = { sign[p(a*|q,k) - 1/|A|] - sign[p(a*|q) - 1/|A|] } / 2
            rewards = (0.5 * torch.sign(gt_answer_probs - 1.0 / torch.tensor(num_ans, device=gt_answer_probs.device).unsqueeze(-1))).squeeze(-1)

            if knowledges_input_ids is not None:
                knowless_results = self.get_reward(
                    questions_input_ids,
                    questions_attention_mask,
                    choicess_input_ids,
                    choicess_attention_mask,
                    choicess_labels,
                    answers_labels,
                    answer_ixs,
                    knowledges_input_ids=None,
                    knowledges_attention_mask=None,
                    override_gain=override_gain,
                    override_bias=override_bias,
                )
                rewards -= knowless_results['rewards/raw']
                knowless_qa_loss = knowless_results['loss/qa']

        gain = self.gain if override_gain is None else override_gain
        bias = self.bias if override_bias is None else override_bias
        rewards_normalized = gain * rewards + bias

        result = {
            'preds': preds, # (B)
            'corrects': corrects, # (B)
            'answer_logitss': answer_logitss, # (B, C)
            'answer_probss': answer_probss, # (B, C)
            'prompts_text': prompts_text,
            'prompts_input_ids': prompts_input_ids,
            'prompts_attention_mask': prompts_attention_mask,
            'rewards/raw': rewards.detach(), # (B)
            'rewards/normalized': rewards_normalized.detach(), # (B)
        }
        if knowledges_input_ids is not None:
            preds_knowless = knowless_results['preds']
            corrects_knowless = knowless_results['corrects']
            rectifieds = corrects.long() - corrects_knowless.long()
            result.update({
                'preds_knowless': preds_knowless, # (B)
                'corrects_knowless': corrects_knowless, # (B)
                'rectifieds': rectifieds, # (B)
            })
        if knowledges_input_ids is not None:
            result.update({
                'loss/qa': knowless_qa_loss,
                'loss/qka': qa_loss,
            })
        else:
            result.update({
                'loss/qa': qa_loss,
            })
        return result

    def kl_penalize_reward(self, results):
        logprobs = results['knowledges_logprobs']
        ref_logprobs = results['knowledges_ref_logprobs']
        mask = results['knowledges_attention_mask']
        normalized_rewards = results['rewards/normalized']

        flattened_rewards = torch.zeros(logprobs.size(), device=normalized_rewards.device, dtype=normalized_rewards.dtype).scatter_(dim=1, index=mask.sum(dim=1, keepdim=True)-1, src=normalized_rewards.unsqueeze(-1))
        kl = logprobs - ref_logprobs
        kl_penalty = self.kl_coef * kl
        penalized_rewards = flattened_rewards - kl_penalty
        # TODO: This is slightly different from the paper

        results['rewards/kl'] = kl # (B, KL)
        results['rewards/kl_penalty'] = kl_penalty # (B, KL)
        results['rewards/penalized'] = penalized_rewards # (B, KL)

    def get_reward_ensemble(self,
                            questions_input_ids: torch.tensor, # (B, QL)
                            questions_attention_mask: torch.tensor, # (B, QL)
                            choicess_input_ids: torch.tensor, # (B, C, AL)
                            choicess_attention_mask: torch.tensor, # (B, C, AL)
                            choicess_labels: torch.tensor, # (B, C, AL)
                            answers_labels: torch.tensor, # (B, AL)
                            answer_ixs: torch.tensor, # (B)
                            knowledgess_input_ids: torch.tensor, # (K, B, KL)
                            knowledgess_attention_mask: torch.tensor, # (K, B, KL)
                            override_gain = None,
                            override_bias = None,
                           ):
        answer_logitsss = [] # [K * (B, C)]
        answer_probsss = [] # [K * (B, C)]

        knowless_results = self.get_reward(
            questions_input_ids, questions_attention_mask,
            choicess_input_ids, choicess_attention_mask, choicess_labels,
            answers_labels, answer_ixs,
            None, None,
            override_gain, override_bias,
            skip_reward=True,
        )
        if not self.no_knowless_expert or len(knowledgess_input_ids) == 0:
            answer_logitsss.append(knowless_results['answer_logitss'])
            answer_probsss.append(knowless_results['answer_probss'])

        for (knowlegdes_input_ids, knowledges_attention_mask) in zip(knowledgess_input_ids, knowledgess_attention_mask):
            results = self.get_reward(
                questions_input_ids, questions_attention_mask,
                choicess_input_ids, choicess_attention_mask, choicess_labels,
                answers_labels, answer_ixs,
                knowlegdes_input_ids, knowledges_attention_mask,
                override_gain, override_bias,
                skip_reward=True,
            )
            answer_logitsss.append(results['answer_logitss'])
            answer_probsss.append(results['answer_probss'])

        answer_logitsss = torch.stack(answer_logitsss) # (1+K, B, C)
        answer_probsss = torch.stack(answer_probsss) # (1+K, B, C)

        if self.ensembling == 'max':
            answer_probss = answer_probsss.max(dim=0).values
            preds = answer_probss.argmax(dim=1)
            selected_knowledge_ixs = answer_probsss.max(dim=-1).values.argmax(dim=0) - 1
        elif self.ensembling == 'moe':
            answer_probss = answer_probsss.mean(dim=0)
            preds = answer_probss.argmax(dim=1)
        elif self.ensembling == 'poe':
            answer_probss = answer_probsss.log().mean(dim=0).exp()
            preds = answer_probss.argmax(dim=1)
        elif self.ensembling == 'majority':
            predss = answer_probsss.argmax(dim=2) # (1+K, B)
            preds = predss.mode(dim=0).values

        # Compute accuracy from argmax answer
        corrects = (preds == answer_ixs)

        preds_knowless = knowless_results['preds']
        corrects_knowless = knowless_results['corrects']
        rectifieds = corrects.long() - corrects_knowless.long()

        return {
            'preds': preds, # (B)
            'corrects': corrects, # (B)
            'preds_knowless': preds_knowless, # (B)
            'corrects_knowless': corrects_knowless, # (B)
            'rectifieds': rectifieds, # (B)
            'answer_logitsss': answer_logitsss,
            'answer_probsss': answer_probsss,
            'answer_probss': answer_probss,
            'selected_knowledge_ixs': selected_knowledge_ixs, # (B) # TODO: Make this compatible with other ensembling methods
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

