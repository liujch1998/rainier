import argparse
import json
import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
import pytorch_lightning as pl
import transformers
device = torch.device('cuda')
pl.seed_everything(19260817)

def init_ds(paths, split):
    ds = []
    for path in paths:
        with open(path) as f:
            js = json.load(f)
        for item in js:
            for k in range(len(item['knowledges'])):
                ds.append({'source': item['query'], 'target': item['knowledges'][k]})
                # for evaluation, only keep the first knowledge for sake of speed
                if split == 'eval':
                    break
    print(f'{split} set size = {len(ds)}')
    return ds

class IterDs(IterableDataset):
    def __init__(self, args):
        super().__init__()
        self.ds = init_ds(args.train_paths, 'train')

    def __iter__(self):
        while True:
            i = np.random.randint(0, len(self.ds)-1)
            yield self.ds[i]

class Ds(Dataset):
    def __init__(self, args):
        super().__init__()
        self.ds = init_ds(args.valid_paths, 'eval')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_type)
        self.args = args

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]

    def collate_fn(self, batch):
        sources = [item['source'] for item in batch]
        sources = self.tokenizer(sources, return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.args.max_input_len)
        input_ids = sources.input_ids
        attention_mask = sources.attention_mask

        targets = [item['target'] for item in batch]
        targets = self.tokenizer(targets, return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.args.max_output_len)
        labels = targets.input_ids
        labels[targets.attention_mask == 0] = -100

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

class GenDs(Dataset):
    def __init__(self, args):
        super().__init__()
        with open(args.input_path) as f:
            self.ds = json.load(f)
            self.ds = self.ds[:args.n]
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_type)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return {'source': item['query']}

    def collate_fn(cls, batch):
        sources = [item['source'] for item in batch]
        sources = self.tokenizer(sources, return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.args.max_input_len)
        input_ids = sources.input_ids
        attention_mask = sources.attention_mask

        return {'input_ids': input_ids, 'attention_mask': attention_mask}

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

def main():
    parser = argparse.ArgumentParser()

    # common
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'gen'])
    parser.add_argument('--model-type', type=str, default='t5-large')
    parser.add_argument('--expr-path', default='../runs/imitation/')
    parser.add_argument('--ckpt-path', default=None)
    parser.add_argument('--max-input-len', type=int, default=256)
    parser.add_argument('--max-output-len', type=int, default=64)

    # train
    parser.add_argument('--train-tasks', type=str, default='obqa,arc_e,arc_h,ai2sci_e,ai2sci_m,csqa,qasc,piqa,siqa,wg')
    parser.add_argument('--max-steps', type=int, default=100000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    
    # gen
    parser.add_argument('--task', type=str)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--m', type=int, default=10)
    parser.add_argument('--top-p', type=float, default=0.5)

    args = parser.parse_args()

    tasks = args.train_tasks.split(',')
    args.train_paths = [f'../data/knowledge/knowledge_gkp_gpt3curie.train.{task}.json' for task in tasks]
    args.valid_paths = [f'../data/knowledge/knowledge_gkp_gpt3curie.dev.{task}.json' for task in tasks]

    if args.ckpt_path is not None:
        model = Model.load_from_checkpoint(args.ckpt_path, args=args)
    else:
        model = Model(args)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        monitor='valid/loss',
        mode='min',
    )
    trainer = pl.Trainer(
        callbacks=[ckpt_callback],
        default_root_dir=args.expr_path,
        deterministic=True,
        accelerator='gpu',
        devices=-1,
        strategy='dp',
        max_steps=args.max_steps,
        val_check_interval=1000,
    )

    if args.mode == 'train':
        train_ds = IterDs(args)
        valid_ds = Ds(args)
        train_dl = DataLoader(train_ds, collate_fn=valid_ds.collate_fn, batch_size=args.batch_size, num_workers=8)
        valid_dl = DataLoader(valid_ds, collate_fn=valid_ds.collate_fn, batch_size=args.batch_size, num_workers=8)

        trainer.fit(model, train_dl, valid_dl)
    elif args.mode == 'gen':
        global test_ds
        test_ds = GenDs(args)
        test_dl = DataLoader(test_ds, collate_fn=GenDs.collate_fn, batch_size=1)

        trainer.logger = None
        trainer.test(model, test_dl, ckpt_path=None) # so to use the model's already loaded weight
        with open(args.output_path, 'w') as f:
            json.dump(test_ds.ds, f, indent=4)

if __name__ == '__main__':
    main()

