import os
os.environ['TRANSFORMERS_CACHE'] = 'cache/'
import transformers
import torch
import pytorch_lightning as pl

class Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.model_type)
        self.args = args

    def forward(self, batch):
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        if self.args.rl:
            logits = outputs.logits # (B, L, V)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            B = logits.size(0)
            L = logits.size(1)
            loss = 0.0
            for b in range(B):
                T = (batch['labels'][b] != -100).sum().item()
                losses = loss_fct(logits[b], batch['labels'][b])
                rl_weights = torch.tensor([self.args.rl_gamma ** (T-t) for t in range(T)], device=device)
                if T < L:
                    rl_weights = torch.cat([rl_weights, torch.zeros((L-T), device=device)])
                loss += torch.dot(losses, rl_weights) * batch['deltas'][b] / T
            loss /= B
            return loss
        if self.args.unlikelihood:
            logits = outputs.logits # (B, L, V)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            B = logits.size(0)
            loss = 0.0
            uloss = 0.0
            for b in range(B):
                if batch['deltas'][b] > 0:
                    loss += loss_fct(logits[b], batch['labels'][b])
                else:
                    uloss += self.args.unlikelihood_alpha * unlikelihood_loss(logits[b], batch['labels'][b])
            loss += uloss
            loss /= B
            return loss
        loss = outputs.loss
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log('train/loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log('valid/loss', loss.item())
        return loss

    def test_step(self, batch, batch_idx):
        global tokenizer
        global test_ds
        outputs = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            do_sample=True,
            num_return_sequences=self.args.m,
            top_p=self.args.top_p,
        )
        knowledges = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        knowledges = list(set(knowledges))
        test_ds.ds[batch_idx]['knowledges'] = knowledges

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)

def load_lightning_model(checkpoint_path, model_type = "t5-large"):
    from argparse import Namespace
    lightning_args = Namespace(model_type = model_type)
    lightning_model = Model.load_from_checkpoint(checkpoint_path = checkpoint_path, args=lightning_args).model
    return lightning_model

if __name__ == "__main__":
    model = load_lightning_model("/gscratch/xlab/liujc/cc-gkm/experiment/kg_ftt5-large/lightning_logs/version_4/checkpoints/epoch=0-step=19999.ckpt", "t5-large")
    from IPython import embed
    embed()