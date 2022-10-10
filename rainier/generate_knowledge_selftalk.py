import os
from tqdm import tqdm
import argparse
import json
from generate_knowledge_gkp import request
from data import datapath_by_task_and_split
import openai
openai.api_key = os.environ['OPENAI_API_KEY']

prefixes = [
    ('What is the definition of', 'The definition of _ is'),
    ('What is the main purpose of', 'The purpose of _ is to'),
    ('What is the main function of a', 'The main function of a _ is'),
    ('What are the properties of a', 'The properties of a _ are that'),
    ('What is a', '_ is'),
    ('What happened as a result of', 'As a result of _,'),
    ('What might have caused', 'The cause of _ was'),
    ('What is a part of', 'A part of _ is'),
    ('What is an example of', 'An example of _ is'),
    ('How would you', 'One would _ by'),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['obqa', 'arc_e', 'arc_h', 'ai2sci_e', 'ai2sci_m', 'csqa', 'qasc', 'piqa', 'siqa', 'wg', 'numersense', 'riddlesense', 'quartz', 'hellaswag'])
    parser.add_argument('--split', type=str, required=True, choices=['train', 'dev'])
    parser.add_argument('--engine', type=str, default='curie', choices=['ada', 'babbage', 'curie', 'davinci'])
    parser.add_argument('--n', type=int, default=None, help='If specified, only run on the first n instances')
    args = parser.parse_args()

    datapath_by_split = datapath_by_task_and_split[args.task]
    datapath = datapath_by_split[args.split if args.split in datapath_by_split else 'default']
    input_path = f'../data/{datapath}/{args.split}.tsv'
    ds = load_tsv(input_path)
    ds = ds[:args.n]

    for item in tqdm(ds):
        knowledges = []
        query = item['query'].split('\\n')[0].strip(' ')
        for (qp, ap) in prefixes:
            qcs = request(
                f'{query} {qp}' ,
                n=1,
                top_p=0.2,
                max_tokens=6,
                stop='?',
                engine='curie',
            )
            for qc in qcs:
                acs = request(
                    f'{query} {qp} {qc} {ap.replace("_", qc)}',
                    n=1,
                    top_p=0.5,
                    max_tokens=10,
                    stop='.',
                    engine='curie',
                )
                for ac in acs:
                    if ac[-1] != '.':
                        ac += '.'
                    knowledges.append(f'{ap.replace("_", qc)}, {ac}')
        item['knowledges'] = list(set(knowledges))

    output_path = f'../data/knowledge/knowledge_selftalk_gpt3{args.engine}.{args.split}.{args.task}.json'
    with open(output_path, 'w') as f:
        json.dump(ds, f, indent=4)

if __name__ == '__main__':
    main()

