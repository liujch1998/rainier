from typing import Union, List, Dict
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from model.t5 import T5ForConditionalGenerationAndTokenRegression
from utils.utils import logits_to_entropy, mask_pad


class Policy:

    def __init__(self,
                 model_type: str,
                 model_ckpt: str,
                 tokenizer,
                 policy_value_sharing: bool,
                 policy_reward_sharing: bool,
                 device,
                 accelerator,
                ):
        self.tokenizer = tokenizer
        self.policy_value_sharing = policy_value_sharing
        self.policy_reward_sharing = policy_reward_sharing
        self.device = device
        self.accelerator = accelerator

        if policy_value_sharing:
            self.model = T5ForConditionalGenerationAndTokenRegression.from_pretrained(model_type)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(model_type)
        if model_ckpt is not None:
            checkpoint = torch.load(model_ckpt, map_location='cpu')
            self.model.load_state_dict(checkpoint, strict=False)
            checkpoint.clear()
        self.model = self.accelerator.prepare(self.model)
        self.model.eval()

    def sample(self,
               questions_input_ids: torch.Tensor, # (B, QL)
               questions_attention_mask: torch.Tensor, # (B, QL)
               sample: bool = True,
               top_k: int = None,
               top_p: float = None,
               temperature: float = None,
              ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        questions_text = self.tokenizer.batch_decode(questions_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        if not self.policy_reward_sharing:
            knowledges_input_ids = self.model.module.generate(
                input_ids=questions_input_ids,
                attention_mask=questions_attention_mask,
                max_length=self.tokenizer.max_knowledge_len + 1,
                min_length=2,
                do_sample=sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            ) # begins with 0 ([BOS]); ends with 1 ([EOS])
            knowledges_input_ids = knowledges_input_ids[:, 1:].contiguous() # no beginning; ends with 1 ([EOS])
        else:
            knowledges_input_ids = self.model.module.generate(
                input_ids=questions_input_ids,
                attention_mask=questions_attention_mask,
                max_length=self.tokenizer.max_knowledge_len + 3,
                min_length=4,
                do_sample=sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                forced_decoder_ids=[[1, 16113], [2, 10]], # 'Knowledge:' is 2 tokens under T5Tokenizer
            ) # begins with 0 ([BOS]); ends with 1 ([EOS])
            knowledges_input_ids = knowledges_input_ids[:, 3:].contiguous() # no beginning; ends with 1 ([EOS])
        knowledges_input_ids = F.pad(knowledges_input_ids, (0, self.tokenizer.max_knowledge_len - knowledges_input_ids.size(1)), value=self.tokenizer.pad_token_id) # (B, KL)
        knowledges_attention_mask = (knowledges_input_ids != self.tokenizer.pad_token_id).int()
        knowledges_text = self.tokenizer.batch_decode(knowledges_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        lowercased_knowledges = [knowledge.lower() for knowledge in knowledges_text]
        lowercased_knowledges_tok = self.tokenizer.batch_encode_plus(
            lowercased_knowledges,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.tokenizer.max_knowledge_len)
        lowercased_knowledges_input_ids = lowercased_knowledges_tok.input_ids
        lowercased_knowledges_attention_mask = lowercased_knowledges_tok.attention_mask

        with torch.no_grad():
            outputs = self.forward_pass(
                questions_input_ids,
                questions_attention_mask,
                knowledges_input_ids,
                knowledges_attention_mask,
            )
        knowledges_logits = outputs['knowledges_logits']
        knowledges_logprobs = outputs['knowledges_logprobs']
        knowledges_entropy = outputs['knowledges_entropy']

        return {
            'questions_text': questions_text,
            'questions_input_ids': questions_input_ids, # (B, QL)
            'questions_attention_mask': questions_attention_mask, # (B, QL)
            'knowledges_text': knowledges_text,
            'knowledges_input_ids': knowledges_input_ids, # (B, KL)
            'knowledges_attention_mask': knowledges_attention_mask, # (B, KL)
            'lowercased_knowledges_text': lowercased_knowledges,
            'lowercased_knowledges_input_ids': lowercased_knowledges_input_ids, # (B, KL)
            'lowercased_knowledges_attention_mask': lowercased_knowledges_attention_mask, # (B, KL)
            'knowledges_logits': knowledges_logits, # (B, KL, V)
            'knowledges_logprobs': knowledges_logprobs, # (B, KL)
            'knowledges_entropy': knowledges_entropy, # (B, KL)
        }

    def forward_pass(self,
                     questions_input_ids: torch.Tensor, # (B, QL)
                     questions_attention_mask: torch.Tensor, # (B, QL)
                     knowledges_input_ids: torch.Tensor, # (B, KL)
                     knowledges_attention_mask: torch.Tensor, # (B, KL)
                    ):

        outputs = self.model(
            input_ids=questions_input_ids,
            attention_mask=questions_attention_mask,
            labels=mask_pad(knowledges_input_ids, knowledges_attention_mask, -100),
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        knowledges_logits = outputs.logits # (B, KL, V)
        logprobs = F.log_softmax(knowledges_logits, dim=-1)
        knowledges_logprobs = torch.gather(logprobs, 2, knowledges_input_ids[:, :, None]).squeeze(2) # (B, KL)
        knowledges_entropy = logits_to_entropy(knowledges_logits) # (B, KL)

        return {
            'knowledges_logits': knowledges_logits, # (B, KL, V)
            'knowledges_logprobs': mask_pad(knowledges_logprobs, knowledges_attention_mask), # (B, KL)
            'knowledges_entropy': mask_pad(knowledges_entropy, knowledges_attention_mask), # (B, KL)
        }
