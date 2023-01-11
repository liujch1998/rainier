import argparse
from collections import defaultdict
from datetime import datetime
from itertools import chain
import json
import logging
import numpy as np
import os
import random
from tqdm import tqdm

import openai
import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import transformers
import wandb

from utils.utils import ensure_dir, set_seed, reduce_mean
from data import QADataset

# logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO'))
# log = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.CRITICAL)

def get_args():
    parser = argparse.ArgumentParser()

    # common
    models = [
        'allenai/unifiedqa-t5-small',
        'allenai/unifiedqa-t5-base',
        'allenai/unifiedqa-t5-large',
        'allenai/unifiedqa-t5-3b',
        'allenai/unifiedqa-t5-11b',
        'allenai/unifiedqa-v2-t5-small-1251000',
        'allenai/unifiedqa-v2-t5-base-1251000',
        'allenai/unifiedqa-v2-t5-large-1251000',
        'allenai/unifiedqa-v2-t5-3b-1251000',
        'allenai/unifiedqa-v2-t5-11b-1251000',
        'openai/ada',
        'openai/babbage',
        'openai/curie',
        'openai/davinci',
        'openai/text-ada-001',
        'openai/text-babbage-001',
        'openai/text-curie-001',
        'openai/text-davinci-002',
        'openai/text-davinci-003',
    ]
    parser.add_argument('--model_type', type=str, required=True, choices=models)
    parser.add_argument('--max_input_len', type=int, default=256)

    parser.add_argument('--data_path', type=str, default='../data/{datapath}/{split}.tsv')

    # train
    parser.add_argument('--train_tasks', type=str, default='obqa,arc_e,arc_h,ai2sci_e,ai2sci_m,csqa,qasc,piqa,siqa,wg')
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()
    return args

args = get_args()

eval_dataset = QADataset('dev', args.train_tasks, args.data_path)
eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=QADataset.collate_fn)

class OpenaiProcessor:
    def __init__(self):
        openai.api_key = os.environ['OPENAI_API_KEY']

        self.engine = args.model_type.replace('openai/', '')

        self.prompt_template = 'Question: {question}\nAnswer:'

    def get_logits_of_last_token(self, prompts):
        while True:
            try:
                response = openai.Completion.create(
                    engine=self.engine,
                    prompt=prompts,
                    max_tokens=0, # suppress continuation
                    logprobs=0, # we don't need logprobs of alternative tokens
                    echo=True, # so the logprobs of prompt tokens are shown
                )
                break
            except Exception as e:
                print(e)
                import time
                time.sleep(60)
        logprobs = [response['choices'][i]['logprobs']['token_logprobs'][-1] for i in range(len(prompts))]
        logprobs = torch.tensor(logprobs)
        return logprobs

    def get_logits_of_answer(self, prompts):
        def reformat(q):
            for i in range(26):
                c = chr(ord('A') + i)
                q = q.replace(f'\n{c}.', f'\n({c})')
            return q
        prompts = [reformat(q) for q in prompts]
        while True:
            try:
                response = openai.Completion.create(
                    engine=self.engine,
                    prompt=prompts,
                    max_tokens=0, # suppress continuation
                    logprobs=0, # we don't need logprobs of alternative tokens
                    echo=True, # so the logprobs of prompt tokens are shown
                )
                break
            except Exception as e:
                print(e)
                import time
                time.sleep(60)
        logits = []
        for i in range(len(prompts)):
            logprobs = response['choices'][i]['logprobs']
            text_offset = logprobs['text_offset']
            offset = prompts[i].index('Answer:') + len('Answer:')
            ix = text_offset.index(offset)
            logprobs = logprobs['token_logprobs'][ix:]
            logprob = np.mean(logprobs)
            logits.append(logprob)
        logits = torch.tensor(logits)
        return logits

    def get_scoress(self, questions, choicess):
        def reformat(q):
            if q.count(' \\n ') == 2:
                question, choices, context = q.split(' \\n ')
                q = f'{context} {question} \\n {choices}'
            q = q.split(' \\n ')[0]
            q = q.replace(' \\n ', '\n')
            q = q.replace('(A)', 'A.')
            for i in range(26):
                c = chr(ord('A') + i)
                q = q.replace(f' ({c})', f'\n{c}.')
            return q
        questions = [reformat(q) for q in questions]
        prompts = []
        for question, choices in zip(questions, choicess):
            # prompts += [f'Question: {question}\nAnswer: {chr(ord("A") + i)}' for i in range(len(choices))]
            prompts += [f'Question: {question}\nAnswer: {choice}' for choice in choices]
        # logits = self.get_logits_of_last_token(prompts)
        logits = self.get_logits_of_answer(prompts)
        logits = logits.tolist()
        scoress = []
        ptr = 0
        for choices in choicess:
            scoress.append(logits[ptr:ptr+len(choices)])
            ptr += len(choices)
        return scoress

class UqaProcessor:
    def __init__(self):
        self.device = torch.device('cuda:0')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('t5-large')
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.model_type)
        self.model.to(self.device)
        self.model.eval()

        self.pos_label = 'yes'
        self.neg_label = 'no'
        pos_ids = self.tokenizer(self.pos_label).input_ids
        neg_ids = self.tokenizer(self.neg_label).input_ids
        assert len(pos_ids) == 2 # the second token is </s> (1)
        assert len(neg_ids) == 2
        self.pos_id = pos_ids[0]
        self.neg_id = neg_ids[0]

    def get_scores(self, sources):
        B = len(sources)
        sources = [_.lower() for _ in sources]
        tok = self.tokenizer(
            sources,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=args.max_input_len).to(self.device)
        with torch.no_grad():
            logits = self.model(
                input_ids=tok.input_ids,
                attention_mask=tok.attention_mask,
                decoder_input_ids=torch.zeros((B, 1), dtype=torch.long, device=self.device),
            ).logits # (B, 1, V)
        pos_logits = logits[:, 0, pos_id] # (B)
        neg_logits = logits[:, 0, neg_id] # (B)
        posneg_logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits.unsqueeze(-1)], dim=1) # (B, 2)
        scores = torch.nn.functional.softmax(posneg_logits, dim=1)[:, 0] # (B)
        scores = scores.tolist()
        return scores

if 'flan' in args.model_type:
    processor = FlanProcessor()
elif 'openai' in args.model_type:
    processor = OpenaiProcessor()

all_scoress = []
all_answer_ixs = []
all_tasks = []
for i, batch in enumerate(tqdm(eval_dataloader)):
    scoress = processor.get_scoress(batch['question'], batch['choices'])
    all_scoress += scoress
    all_answer_ixs += batch['answer_ix']
    all_tasks += batch['task']

corrects = []
corrects_by_task = defaultdict(list)
for scores, answer_ix, task in zip(all_scoress, all_answer_ixs, all_tasks):
    correct = (answer_ix == np.argmax(scores))
    corrects.append(correct)
    corrects_by_task[task].append(correct)

acc_weighted = np.mean(corrects)
acc_by_task = {k: np.mean(v) for k, v in corrects_by_task.items()}
acc_unweighted = np.mean(list(acc_by_task.values()))

print(f'acc_weighted = {acc_weighted:.4f} | acc_unweighted = {acc_unweighted:.4f}')
print('Accuracy by task:')
for task, acc in acc_by_task.items():
    print(f'\t{task} = {acc:.4f}')
print('\n'.join([f'{acc*100:.2f}' for task, acc in acc_by_task.items()]))

