import torch
from transformers import T5Tokenizer
from model.t5 import T5ForTokenRegression
from utils.utils import mask_pad


class Value:

    def __init__(self,
                 model_type,
                 model_ckpt,
                 device,
                 device_map,
                 model=None,
                ):
        self.tokenizer = T5Tokenizer.from_pretrained(model_type)

        if model is not None:
            self.model = model
            self.device = device
            return

        self.model = T5ForTokenRegression.from_pretrained(model_type)
        if model_ckpt is not None:
            checkpoint = torch.load(model_ckpt, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint, strict=False)
            checkpoint.clear()
        self.model.eval()
        self.model.to(device)
        self.model.encoder.parallelize(device_map=device_map)
        self.model.decoder.parallelize(device_map=device_map)
        self.model.model_parallel = True

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.device = device

    def forward_pass(self,
                     query_input_ids: torch.Tensor,
                     query_mask: torch.Tensor,
                     response_input_ids: torch.Tensor,
                     response_mask: torch.Tensor,
                    ):

        query_input_ids = query_input_ids.to(self.device)
        query_mask = query_mask.to(self.device)
        response_input_ids = response_input_ids.to(self.device)
        response_mask = response_mask.to(self.device)

        outputs = self.model.forward_cls(
            input_ids=query_input_ids,
            attention_mask=query_mask,
            labels=mask_pad(response_input_ids, response_mask, -100),
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        return {
            'response/value': mask_pad(outputs.logits, response_mask)
        }

