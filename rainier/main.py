import os
import json
from datetime import datetime
import logging
import itertools
from tqdm import tqdm
import math
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from args import get_args
from utils.utils import ensure_dir, ceil_div
from data import QADataset
from model.policy import Policy
from model.value import Value
from model.reward import Reward
from ppo import PPOTrainer

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO'))
log = logging.getLogger(__name__)


def main():
    args = get_args()

    # Random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
    if args.mode == 'train':
        if num_gpus == 8:  # 8x RTX6000
            device_map = {
                0: [0],
                1: [1, 2, 3],
                2: [4, 5, 6],
                3: [7, 8, 9],
                4: [10, 11 ,12],
                5: [13, 14, 15],
                6: [16, 17, 18, 19],
                7: [20, 21, 22, 23],
            }
        else:
            log.error('Invalid number of GPUs! Please use 8')
            exit(-1)
    elif args.mode == 'eval':
        if num_gpus == 4:  # 4x RTX6000
            device_map = {
		0: [0],
                1: [1, 2, 3, 4, 5, 6, 7],
                2: [8, 9, 10, 11, 12, 13, 14, 15],
                3: [16, 17, 18, 19, 20, 21, 22, 23],
            }

    # Set up save directories
    if args.mode == 'train':
        args.output_dir = '../runs'
        if args.load_from_ckpt is not None:
            args.save_dir = os.path.dirname(os.path.dirname(args.load_from_ckpt))
        else:
            time = datetime.now()
            date_time = time.strftime('%b%d_%H-%M-%S')
            import socket
            args.save_dir = os.path.join(args.output_dir, date_time + '_' + socket.gethostname())
        args.reward_dir = os.path.join(args.save_dir, 'reward')
        args.model_dir = os.path.join(args.save_dir, 'model')
        args.tensorboard_dir = os.path.join(args.save_dir, 'tensorboard')
        args.knowledge_dir = os.path.join(args.save_dir, 'knowledge')
        args.inference_dir = os.path.join(args.save_dir, 'inference')
        for d in [args.save_dir, args.reward_dir, args.model_dir, args.tensorboard_dir, args.knowledge_dir, args.inference_dir]:
            ensure_dir(d)

    elif args.mode == 'eval':
        if args.load_from_ckpt is not None:
            args.save_dir = os.path.dirname(os.path.dirname(args.load_from_ckpt))
            args.save_dir = args.save_dir.replace('runs/', 'eval/')
            ckp = args.load_from_ckpt.split('ckp_')[-1].strip('.pth')
            args.save_dir += f'_ckp-{ckp}'
        elif args.eval_ckpt is not None:
            args.save_dir = os.path.dirname(args.eval_ckpt)
        else:
            log.error('You must provide either --ckpt or --load_from_ckpt!')
            exit(-1)
        args.tensorboard_dir = os.path.join(args.save_dir, 'tensorboard')
        args.knowledge_dir = os.path.join(args.save_dir, 'knowledge')
        args.inference_dir = os.path.join(args.save_dir, 'inference')
        for d in [args.save_dir, args.tensorboard_dir, args.knowledge_dir, args.inference_dir]:
            ensure_dir(d)

    log.info(f'Write to output directory: {args.save_dir}')
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Load data
    log.info(f'Loading data ...')

    if args.mode == 'train':
        train_dataset = QADataset('train', args.train_tasks)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=QADataset.collate_fn)
        log.info(f'Loaded train set with {len(train_dataset)} instances')

        eval_dataset = QADataset('dev', args.train_tasks)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=QADataset.collate_fn)
        log.info(f'Loaded dev set with {len(eval_dataset)} instances')

    elif args.mode == 'eval':
        train_dataset = None
        train_dataloader = None

        eval_dataset = QADataset(args.eval_split, args.eval_tasks)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=QADataset.collate_fn)
        log.info(f'Loaded {args.eval_split} set with {len(eval_dataset)} instances')


    # Initialize models and optimizer
    log.info(f'Initializing models ...')

    if args.mode == 'train':
        ref_policy = Policy(
            model_type=args.model_type,
            model_ckpt=args.model_ckpt,
            policy_value_sharing=args.policy_value_sharing,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            device=devices[0],
            device_map=device_map,
        )
        policy = Policy(
            model_type=args.model_type,
            model_ckpt=args.model_ckpt,
            policy_value_sharing=args.policy_value_sharing,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            device=devices[0],
            device_map=device_map,
        )
        # TODO: Try initializing this with model_ckpt as well
        value = Value(
            model_type=args.model_type,
            model_ckpt=args.model_ckpt if args.use_model_ckpt_for_value else None,
            model=policy.model if args.policy_value_sharing else None,
            device=devices[0],
            device_map=device_map,
        )
        reward = Reward(
            model_type=args.qa_model_type,
            model_ckpt=args.qa_model_ckpt,
            max_input_len=args.max_input_len,
            batch_size=args.batch_size,
            reward_shape=args.reward_shape,
            kl_coef=args.kl_coef,
            ensembling=args.ensembling,
            device=devices[0],
        )

        optimizer = torch.optim.Adam(policy.model.parameters() if args.policy_value_sharing else itertools.chain(policy.model.parameters(), value.model.parameters()), lr=args.lr, eps=1e-5)
        args.total_steps = ceil_div(args.total_episodes, args.batch_size)
        warmup_steps = math.ceil(args.num_warmup_step_ratio * args.total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.total_steps)
        step_num = 0

        # Load from checkpoint if continue training
        if args.load_from_ckpt is not None:
            checkpoint = torch.load(args.load_from_ckpt)
            policy.model.load_state_dict(checkpoint['policy_model'])
            value.model.load_state_dict(checkpoint['value_model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            step_num = checkpoint['step']
            checkpoint.clear()

            # Reuse the reward normalization results
            reward.read_reward_norm(args.reward_dir)

    elif args.mode == 'eval':
        ref_policy = None
        policy = Policy(
            model_type=args.model_type,
            model_ckpt=args.model_ckpt,
            policy_value_sharing=args.policy_value_sharing,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            device_map=device_map,
            device=devices[0],
        )
        value = None
        reward = Reward(
            model_type=args.qa_model_type,
            model_ckpt=args.qa_model_ckpt,
            max_input_len=args.max_input_len,
            batch_size=args.batch_size,
            reward_shape=args.reward_shape,
            kl_coef=args.kl_coef,
            ensembling=args.ensembling,
            device=devices[0],
        )

        optimizer = None
        scheduler = None

        if args.load_from_ckpt is not None:
            checkpoint = torch.load(args.load_from_ckpt, map_location=torch.device('cpu'))
            policy.model.load_state_dict(checkpoint['policy_model'])
            step_num = checkpoint['step']
            checkpoint.clear()
        elif args.eval_ckpt is not None:
            checkpoint = torch.load(args.eval_ckpt, map_location=torch.device('cpu'))
            policy.model.load_state_dict(checkpoint)
            checkpoint.clear()
            step_num = 0

    # Set up trainer
    trainer = PPOTrainer(
        args=args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        policy_model=policy,
        ref_policy_model=ref_policy,
        value_model=value,
        reward_model=reward,
        optimizer=optimizer,
        scheduler=scheduler,
        log=log,
        log_dir=args.tensorboard_dir,
    )

    # Normalize the rewards to so that initially they have mean 0, var 1
    if args.mode == 'train':
        if args.load_from_ckpt is None:
            log.info('Setting reward norm')
            if args.gain is not None and args.bias is not None:
                reward.gain = args.gain
                reward.bias = args.bias
            else:
                trainer.set_reward_norm()
            log.info(f'Set reward norm as gain = {reward.gain}, bias = {reward.bias}')
            reward.write_reward_norm(args.reward_dir)

    # Evaluate baseline (no knowledge)
    if args.eval_baseline:
        trainer.eval(step=-1)

    # Train or evaluate
    if args.mode == 'train':
        pbar = tqdm(list(range(step_num, args.total_steps)))
        for step_num in pbar:
            trainer.train(step_num)
    elif args.mode == 'eval':
        trainer.eval(step=step_num)


if __name__ == '__main__':
    main()

