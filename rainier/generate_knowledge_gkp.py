import os
from tqdm import tqdm
import argparse
import json
from data import datapath_by_task_and_split
import openai
openai.api_key = os.environ['OPENAI_API_KEY']

def request(
    prompt: str,
    engine='curie',
    max_tokens=64,
    temperature=1.0,
    top_p=1.0,
    n=1,
    stop='\n',
    presence_penalty=0.0,
    frequency_penalty=0.0,
):  
    # retry request (handles connection errors, timeouts, and overloaded API)
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )   
            break
        except Exception as e:
            tqdm.write(str(e))
            tqdm.write("Retrying...")
            import time
            time.sleep(60)

    generations = [gen['text'].lstrip() for gen in response['choices']]
    generations = [_ for _ in generations if _ != ''] 
    return generations

def load_tsv(path):
    ds = []
    with open(path) as f:
        for line in f:
            try:
                [query, answer] = line.strip('\n').split('\t')
            except Exception as e:
                continue
            query = query.strip(' ')
            answer = answer.strip(' ')
            ds.append({ 'query': query, 'answer': answer })
    return ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['obqa', 'arc_e', 'arc_h', 'ai2sci_e', 'ai2sci_m', 'csqa', 'qasc', 'piqa', 'siqa', 'wg', 'numersense', 'riddlesense', 'quartz', 'hellaswag'])
    parser.add_argument('--split', type=str, required=True, choices=['train', 'dev'])
    parser.add_argument('--engine', type=str, default='curie', choices=['ada', 'babbage', 'curie', 'davinci'])
    parser.add_argument('--m', type=int, default=20, help='Number of knowledge to generate per instance')
    parser.add_argument('--top-p', type=float, default=0.5)
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--n', type=int, default=None, help='If specified, only run on the first n instances')
    args = parser.parse_args()

    datapath_by_split = datapath_by_task_and_split[args.task]
    datapath = datapath_by_split[args.split if args.split in datapath_by_split else 'default']
    input_path = f'../data/{datapath}/{args.split}.tsv'
    ds = load_tsv(input_path)
    ds = ds[:args.n]

    prompt_path = f'../data/prompts/{args.task}.txt'
    with open(prompt_path) as f:
        prompt_template = f.read().strip('\n')

    for item in tqdm(ds):
        prompt = prompt_template.replace('{question}', item['query'])
        knowledges = request(
            prompt,
            engine=args.engine,
            n=args.m,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        item['knowledges'] = knowledges # keep duplicates

    output_path = f'../data/knowledge/knowledge_gkp_gpt3{args.engine}.{args.split}.{args.task}.json'
    with open(output_path, 'w') as f:
        json.dump(ds, f, indent=4)

if __name__ == '__main__':
    main()

