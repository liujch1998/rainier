import json
import os
import random
from torch.utils.data import Dataset

datapath_by_task_and_split = {
    'obqa': {'default': 'uqa/openbookqa'},
    'arc_e': {'default': 'uqa/arc_easy'},
    'arc_h': {'default': 'uqa/arc_hard'},
    'ai2sci_e': {'default': 'uqa/ai2_science_elementary'},
    'ai2sci_m': {'default': 'uqa/ai2_science_middle'},
    'csqa': {'default': 'uqa/commonsenseqa'},
    'qasc': {'default': 'uqa/qasc'},
    'piqa': {'default': 'uqa/physical_iqa', 'test': 'uqa/physical_iqa_test'},
    'siqa': {'default': 'uqa/social_iqa', 'test': 'uqa/social_iqa_test'},
    'wg': {'default': 'uqa/winogrande_xl'},

    'numersense': {'default': 'numersense'},
    'riddlesense': {'default': 'riddlesense'},
    'quartz': {'default': 'quartz'},
    'hellaswag': {'default': 'hellaswag'},

    'sciq': {'default': 'sciq'},
    'quarel': {'default': 'quarel'},
    'quartz': {'default': 'quartz'},
    'wsc273_': {'default': 'wsc273_'},
    'copa_': {'default': 'copa_'},
    'numersense_': {'default': 'numersense_'},
    'truthfulqa_mc1': {'default': 'truthfulqa_mc1'},
}

'''
tasks_by_split = {
    'train': ['obqa', 'arc_e', 'arc_h', 'ai2sci_e', 'ai2sci_m', 'csqa', 'qasc', 'piqa', 'siqa', 'wg'],
    'valid': ['obqa', 'arc_e', 'arc_h', 'ai2sci_e', 'ai2sci_m', 'csqa', 'qasc', 'piqa', 'siqa', 'wg', 'numersense', 'riddlesense', 'quartz', 'hellaswag'],
    'test': ['obqa', 'arc_e', 'arc_h', 'ai2sci_e', 'ai2sci_m', 'csqa', 'qasc', 'piqa', 'siqa', 'wg', 'numersense', 'riddlesense', 'quartz', 'hellaswag'],
}
'''

# This does not lowercase the data, by default
class QADataset(Dataset):
    def __init__(self, split, tasks, data_path):
        super().__init__()
        self.split = split
        self.tasks = tasks.split(',')
        self.data_path = data_path

        self.instances = self.load_datasets()

        if split == 'train':
            random.shuffle(self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

    def load_datasets(self):
        def parse_choices(s):
            '''
            s: serialized_choices '(A) ... (B) ... (C) ...'
            '''
            choices = []
            key = 'A' if s.find('(A)') != -1 else 'a'
            while True:
                pos = s.find(f'({chr(ord(key) + 1)})')
                if pos == -1:
                    break
                choice = s[3:pos]
                s = s[pos:]
                choice = choice.strip(' ')
                choices.append(choice)
                key = chr(ord(key) + 1)
            choice = s[3:]
            choice = choice.strip(' ')
            choices.append(choice)
            return choices

        instances = []
        if self.data_path.endswith('.tsv'):
            for task in self.tasks:
                skipped = 0
                datapath_by_split = datapath_by_task_and_split[task]
                datapath = datapath_by_split[self.split if self.split in datapath_by_split else 'default']
                with open(self.data_path.replace('{datapath}', datapath).replace('{split}', self.split)) as f:
                    for line in f:
                        try:
                            q, a = line.strip('\n').split('\t')
                            q = q.strip(' ')
                            a = a.strip(' ')
                            choices = parse_choices(q.split('\\n')[1].strip(' '))
                            answer_ix = choices.index(a)
                        except Exception as e:
                            skipped += 1
                            continue
                        instances.append({
                            'task': task,
                            'question': q,
                            'choices': choices,
                            'answer': a,
                            'answer_ix': answer_ix,
                        })
                print(f'Loaded dataset for task {task} split {self.split}, skipped {skipped} instances')
        elif self.data_path.endswith('.json'):
            for task in self.tasks:
                skipped = 0
                with open(self.data_path.replace('{task}', task).replace('{split}', self.split)) as f:
                    js = json.load(f)
                    for item in js:
                        try:
                            q, a = item['query'], item['answer']
                            choices = parse_choices(q.split('\\n')[1].strip(' '))
                            answer_ix = choices.index(a)
                            knowledges = item['knowledges']
                        except Exception as e:
                            skipped += 1
                            continue
                        instances.append({
                            'task': task,
                            'question': q,
                            'choices': choices,
                            'answer': a,
                            'answer_ix': answer_ix,
                            'knowledges': knowledges,
                        })
                print(f'Loaded dataset for task {task} split {self.split}, skipped {skipped} instances')
        print(f'Loaded split {self.split} with {len(instances)} total instances')
        return instances

    # Make a collate function to fix dataloader weird list batching
    @staticmethod
    def collate_fn(batch):
        batched_dict = {}
        all_keys = batch[0].keys()
        for k in all_keys:
            batched_dict[k] = []
        for cur_dict in batch:
            for k in all_keys:
                batched_dict[k].append(cur_dict[k])
        return batched_dict

