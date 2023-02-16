import torch
from model.t5 import T5ForTokenRegression
from utils.utils import mask_pad


class Value:

    def __init__(self,
                 model_type,
                 model_ckpt,
                 model,
                 tokenizer,
                ):
        self.tokenizer = tokenizer

        if model is not None:
            self.model = model
            return

        self.model = T5ForTokenRegression.from_pretrained(model_type)
        self.model.config.pad_token_id = tokenizer.pad_token_id
        if model_ckpt is not None:
            checkpoint = torch.load(model_ckpt, map_location='cpu')
            self.model.load_state_dict(checkpoint, strict=False)
            checkpoint.clear()
        self.model.eval()

    def forward_pass(self,
                     questions_input_ids: torch.Tensor, # (B, QL)
                     questions_attention_mask: torch.Tensor, # (B, QL)
                     knowledges_input_ids: torch.Tensor, # (B, KL)
                     knowledges_attention_mask: torch.Tensor, # (B, KL)
                    ):

        outputs = self.model.forward_cls(
            input_ids=questions_input_ids,
            attention_mask=questions_attention_mask,
            labels=mask_pad(knowledges_input_ids, knowledges_attention_mask, -100),
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        return {
            'knowledges_value': mask_pad(outputs.logits, knowledges_attention_mask), # (B, KL)
        }
