import argparse
from datetime import datetime
import json
import logging
import numpy as np
import os
import random
from tqdm import tqdm

import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import transformers
import wandb

from utils.utils import ensure_dir, set_seed

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO'))
log = logging.getLogger(__name__)


class Ds(Dataset):
    def __init__(self, paths, split):
        super().__init__()
        self.ds = self.init_ds(paths, split)

    def init_ds(self, paths, split):
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
        if split == 'train':
            random.shuffle(ds)
        print(f'{split} set size = {len(ds)}')
        return ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]

    @staticmethod
    def collate_fn(batch):
        return {k: [item[k] for item in batch] for k in batch[0]}

class Trainer:
    def __init__(self,
                 args,
                 train_dataloader,
                 eval_dataloader,
                 tokenizer,
                 model,
                 optimizer,
                 init_step,
                 eval_losses,
                 device,
                ):
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.train_sampler = iter(self.train_dataloader)
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer
        self.device = device
        if not args.nosave:
            # self.writer = SummaryWriter(log_dir=args.tensorboard_dir)
            wandb.init(project='rainier_stageI', name=args.run_name, config=args)
            wandb.define_metric('train/step')
            wandb.define_metric('eval/step')
            wandb.define_metric('train/loss', step_metric='train/step', summary='min')
            wandb.define_metric('eval/loss', step_metric='eval/step', summary='min')

        self.train_sampler = iter(self.train_dataloader)
        for _ in range((init_step * args.accumulate_grad_batches) % len(self.train_dataloader)):
            next(self.train_sampler)

        self.eval_losses = eval_losses

    def loss(self, batch):
        source_tok = self.tokenizer.batch_encode_plus(
            batch['source'],
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.args.max_input_len).to(self.device)
        target_tok = self.tokenizer.batch_encode_plus(
            batch['target'],
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.args.max_output_len).to(self.device)
        labels = target_tok.input_ids
        labels[target_tok.attention_mask == 0] = -100

        loss = self.model(
            input_ids=source_tok.input_ids,
            attention_mask=source_tok.attention_mask,
            labels=labels,
        ).loss

        return loss

    def train(self, step):
        self.eval(step=step)
        self.save(step=step)

        self.model.train()
        self.optimizer.zero_grad()
        for _ in range(self.args.accumulate_grad_batches):
            try:
                batch = next(self.train_sampler)
            except StopIteration:
                self.train_sampler = iter(self.train_dataloader)
                batch = next(self.train_sampler)
            loss = self.loss(batch)
            loss.backward()
        self.optimizer.step()

        if not self.args.nosave:
            if step % self.args.log_interval == 0:
                # self.writer.add_scalar('train/loss', loss.item(), step)
                wandb.log({'train/loss': loss.item(), 'train/step': step})

    def eval(self, step):
        if step % self.args.eval_interval != 0:
            return
        if step in self.eval_losses:
            return
        log.info(f'Evaluating [step {step}] ...')

        losses = []
        for i, batch in enumerate(tqdm(self.eval_dataloader)):
            self.model.eval()
            with torch.no_grad():
                loss = self.loss(batch)
            losses.append(loss.item())
        loss = np.mean(losses)

        if self.args.nosave:
            self.eval_losses[step] = loss
        else:
            # self.writer.add_scalar('eval/loss', loss, step)
            wandb.log({'eval/loss': loss.item(), 'eval/step': step})

            prev_best_step = None if len(self.eval_losses) == 0 else min(self.eval_losses, key=self.eval_losses.get)
            self.eval_losses[step] = loss
            if prev_best_step is None or loss < self.eval_losses[prev_best_step]:
                if prev_best_step is not None:
                    try:
                        os.remove(f'{self.args.model_dir}/ckp_{prev_best_step}.pth')
                    except:
                        log.warning(f'Cannot remove previous best ckpt!')
                self.save(step, last=False)
                log.info(f'Best ckpt updated to [step {step}]')

    def save(self, step, last=True):
        if self.args.nosave:
            return
        if step % self.args.save_interval != 0:
            return
        # this will overwrite an existing ckpt with the save filename!
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': step,
            'eval_losses': self.eval_losses,
        }, f'{self.args.model_dir}/{"last" if last else "ckp_" + str(step)}.pth')
        log.info(f'[step {step}] model checkpoint saved')

def get_args():
    parser = argparse.ArgumentParser()

    # common
    parser.add_argument('--model_type', type=str, default='t5-large')
    parser.add_argument('--load_from_ckpt', default=None)
    parser.add_argument('--max_input_len', type=int, default=256)
    parser.add_argument('--max_output_len', type=int, default=32)

    # train
    parser.add_argument('--train_tasks', type=str, default='obqa,arc_e,arc_h,ai2sci_e,ai2sci_m,csqa,qasc,piqa,siqa,wg')
    parser.add_argument('--total_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)

    # other
    parser.add_argument(
        '--log_interval', type=int, default=50, help='step interval to log stats')
    parser.add_argument(
        '--save_interval', type=int, default=1000, help='step interval to save model checkpoints')
    parser.add_argument(
        '--eval_interval', type=int, default=1000, help='step interval to do evaluation')
    parser.add_argument('--nosave', default=False, action='store_true')

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    set_seed()

    # GPUs
    num_gpus = torch.cuda.device_count()
    log.info(f'Detected {num_gpus} GPUS')
    devices = {}
    if torch.cuda.is_available():
        for i in range(num_gpus):
            devices[i] = torch.device('cuda:' + str(i))
    else:
        devices[0] = torch.device('cpu')

    device_map = None
    if num_gpus == 4:  # 4x V100 for T5-large
        device_map = {
            0: [0],
            1: [1, 2, 3, 4, 5, 6, 7],
            2: [8, 9, 10, 11, 12, 13, 14, 15],
            3: [16, 17, 18, 19, 20, 21, 22, 23],
        }
    elif num_gpus == 8:  # 8x V100 for T5-3b
        device_map = {
            0: [0],
            1: [1, 2, 3],
            2: [4, 5, 6],
            3: [7, 8, 9],
            4: [10, 11, 12],
            5: [13, 14, 15],
            6: [16, 17, 18, 19],
            7: [20, 21, 22, 23],
        }
    else:
        log.error('Invalid number of GPUs!')
        exit(-1)

    # Set up save directories
    if not args.nosave:
        args.output_dir = '../runs_stageI/'
        if args.load_from_ckpt is not None:
            args.save_dir = os.path.dirname(os.path.dirname(args.load_from_ckpt))
            args.run_name = args.save_dir.split('/')[-1]
        else:
            time = datetime.now()
            date_time = time.strftime('%b%d_%H-%M-%S')
            import socket
            args.run_name = date_time + '_' + socket.gethostname()
            args.save_dir = os.path.join(args.output_dir, args.run_name)
        args.model_dir = os.path.join(args.save_dir, 'model')
        args.tensorboard_dir = os.path.join(args.save_dir, 'tensorboard')
        for d in [args.save_dir, args.model_dir, args.tensorboard_dir]:
            ensure_dir(d)
    
        log.info(f'Write to output directory: {args.save_dir}')
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    # Load data
    log.info(f'Loading data ...')
    tasks = args.train_tasks.split(',')
    train_paths = [f'../data/knowledge/knowledge_gkp_gpt3curie.train.{task}.json' for task in tasks]
    eval_paths = [f'../data/knowledge/knowledge_gkp_gpt3curie.dev.{task}.json' for task in tasks]
    train_dataset = Ds(train_paths, 'train')
    eval_dataset = Ds(eval_paths, 'eval')
    # train ds is shuffled in its constructor
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=Ds.collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=Ds.collate_fn)

    # Initialize models and optimizer
    log.info(f'Initializing models ...')
    tokenizer = transformers.T5Tokenizer.from_pretrained(args.model_type)
    model = transformers.T5ForConditionalGeneration.from_pretrained(args.model_type)
    model.to(devices[0])
    model.parallelize(device_map=device_map)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    init_step = 0
    eval_losses = {}

    # Load from checkpoint if continue training
    if args.load_from_ckpt is not None:
        checkpoint = torch.load(args.load_from_ckpt)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        init_step = checkpoint['step']
        eval_losses = checkpoint['eval_losses']
        checkpoint.clear()

    # Set up trainer
    trainer = Trainer(
        args=args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        model=model,
        optimizer=optimizer,
        init_step=init_step,
        eval_losses=eval_losses,
        device=devices[0],
    )

    # Train
    pbar = tqdm(list(range(init_step, args.total_steps + 1)))
    for step in pbar:
        trainer.train(step)


if __name__ == '__main__':
    main()

