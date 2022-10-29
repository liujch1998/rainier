import os
from typing import Union, List, Dict
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
from model.t5 import T5ForConditionalGenerationAndTokenRegression
from utils.utils import logits_to_entropy, mask_pad


class Policy:

    def __init__(self,
                 model_type: str,
                 model_ckpt: str,
                 policy_value_sharing: bool,
                 max_input_len: int,
                 max_output_len: int,
                 device,
                 device_map = None,
                ):
        self.tokenizer = T5Tokenizer.from_pretrained(model_type)
        if policy_value_sharing:
            self.model = T5ForConditionalGenerationAndTokenRegression.from_pretrained(model_type)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(model_type)
        if model_ckpt is not None:
            checkpoint = torch.load(model_ckpt, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint, strict=False)
            checkpoint.clear()
        self.model.eval()
        self.model.to(device)
        if device != 'cpu':
            self.model.parallelize(device_map=device_map)

        self.policy_value_sharing = policy_value_sharing
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.device = device

    def sample(self,
               text: List[str],
               sample: bool = True,
               top_k: int = None,
               top_p: float = None,
               temperature: float = None,
              ) -> Dict[str, Union[torch.Tensor, List[str]]]:

        tokenized = self.tokenizer.batch_encode_plus(
            text,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.max_input_len)
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)

        response_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_output_len + 1,
            min_length=3,
            do_sample=sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        ) # begins with 0 ([BOS]); ends with 1 ([EOS])
        response_ids = response_ids[:, 1:].contiguous() # no beginning; ends with 1 ([EOS])
        response_mask = (response_ids != self.model.config.pad_token_id).int()
        response_text = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with torch.no_grad():
            outputs = self.forward_pass(input_ids, attention_mask, response_ids, response_mask)
        response_logits = outputs['response/logits']
        response_logprobs = outputs['response/logprobs']
        response_entropy = outputs['response/entropy']

        return {
            'query/text': text,
            'query/input_ids': input_ids,
            'query/mask': attention_mask,
            'response/text': response_text,
            'response/input_ids': response_ids,
            'response/mask': response_mask,
            'response/logits': response_logits,
            'response/logprobs': response_logprobs,
            'response/entropy': response_entropy,
        }

    def forward_pass(self,
                     query_input_ids: torch.Tensor,
                     query_mask: torch.Tensor,
                     response_input_ids: torch.Tensor,
                     response_mask: torch.Tensor,
                    ):

        outputs = self.model(
            input_ids=query_input_ids,
            attention_mask=query_mask,
            labels=mask_pad(response_input_ids, response_mask, -100),
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        response_logits = outputs.logits # (B, RL, V)
        logprobs = F.log_softmax(response_logits, dim=-1)
        response_logprobs = torch.gather(logprobs, 2, response_input_ids[:, :, None]).squeeze(2) # (B, RL)
        response_entropy = logits_to_entropy(response_logits) # (B, RL)

        return {
            'response/logits': response_logits,
            'response/logprobs': mask_pad(response_logprobs, response_mask),
            'response/entropy': mask_pad(response_entropy, response_mask),
        }

