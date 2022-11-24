import os, sys
sys.path.append('../rainier')
from flask import Flask, render_template, redirect, request, jsonify, make_response
import json
import numpy as np
import random

import torch
import transformers

from utils.utils import reduce_mean, set_seed

device = torch.device('cuda')

max_input_len = 256
max_output_len = 32
m = 10
top_p = 0.5

'''
def set_seed(seed=19260817, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
'''

class InteractiveRainier:

    def __init__(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('allenai/unifiedqa-t5-large')
        self.rainier_model = transformers.AutoModelForSeq2SeqLM.from_pretrained('liujch1998/rainier-large').to(device)
        self.qa_model = transformers.AutoModelForSeq2SeqLM.from_pretrained('allenai/unifiedqa-t5-large').to(device)
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100,reduction='none')

    def parse_choices(self, s):
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

    def run(self, question):
        set_seed()

        tokenized = self.tokenizer(question, return_tensors='pt', padding='max_length', truncation='longest_first', max_length=max_input_len).to(device) # (1, L)
        knowledges_ids = self.rainier_model.generate(
            input_ids=tokenized.input_ids,
            max_length=max_output_len + 1,
            min_length=3,
            do_sample=True,
            num_return_sequences=m,
            top_p=top_p,
        ) # (K, L); begins with 0 ([BOS]); ends with 1 ([EOS])
        knowledges_ids = knowledges_ids[:, 1:].contiguous() # no beginning; ends with 1 ([EOS])
        knowledges = self.tokenizer.batch_decode(knowledges_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        knowledges = list(set(knowledges))
        knowledges = [''] + knowledges

        prompts = [question + (f' \\n {knowledge}' if knowledge != '' else '') for knowledge in knowledges]
        choices = self.parse_choices(question)
        prompts = [prompt.lower() for prompt in prompts]
        choices = [choice.lower() for choice in choices]
        answer_logitss = []
        for choice in choices:
            tokenized_prompts = self.tokenizer(prompts, return_tensors='pt', padding='max_length', truncation='longest_first', max_length=max_input_len).to(device) # (1+K, L)
            tokenized_choices = self.tokenizer([choice], return_tensors='pt', padding='max_length', truncation='longest_first', max_length=max_input_len).to(device) # (1, L)
            pad_mask = (tokenized_choices.input_ids == self.tokenizer.pad_token_id)
            tokenized_choices.input_ids[pad_mask] = -100
            tokenized_choices.input_ids = tokenized_choices.input_ids.repeat(len(knowledges), 1) # (1+K, L)

            with torch.no_grad():
                logits = self.qa_model(
                    input_ids=tokenized_prompts.input_ids,
                    labels=tokenized_choices.input_ids,
                ).logits # (1+K, L, V)

            losses = self.loss_fct(logits.view(-1, logits.size(-1)), tokenized_choices.input_ids.view(-1))
            losses = losses.view(tokenized_choices.input_ids.shape) # (1+K, L)
            losses = reduce_mean(losses, ~pad_mask, axis=-1) # (1+K)
            answer_logitss.append(-losses)
        answer_logitss = torch.stack(answer_logitss, dim=1) # (1+K, C)
        answer_probss = answer_logitss.softmax(dim=1) # (1+K, C)

        # Ensemble
        knowless_pred = answer_probss[0, :].argmax(dim=0).item()
        knowless_pred = choices[knowless_pred]

        answer_probs = answer_probss.max(dim=0).values # (C)
        knowful_pred = answer_probs.argmax(dim=0).item()
        knowful_pred = choices[knowful_pred]
        selected_knowledge_ix = answer_probss.max(dim=1).values.argmax(dim=0).item()
        selected_knowledge = knowledges[selected_knowledge_ix]

        return {
            'question': question,
            'knowledges': knowledges,
            'knowless_pred': knowless_pred,
            'knowful_pred': knowful_pred,
            'selected_knowledge': selected_knowledge,
        }

app = Flask(__name__)
debug = False
if not debug:
    rainier = InteractiveRainier()

# These examples are copied from Table 5 of the Rainier paper
questions = [
    'Sydney rubbed Addison’s head because she had a horrible headache. What will happen to Sydney? \\n (A) drift to sleep (B) receive thanks (C) be reprimanded',
    'What would vinyl be an odd thing to replace? \\n (A) pants (B) record albums (C) record store (D) cheese (E) wallpaper',
    'Some pelycosaurs gave rise to reptile ancestral to \\n (A) lamphreys (B) angiosperm (C) mammals (D) paramecium (E) animals (F) protozoa (G) arachnids (H) backbones',
    'Adam always spent all of the free time watching Tv unlike Hunter who volunteered, due to _ being lazy. \\n (A) Adam (B) Hunter',
    'Causes bad breath and frightens blood-suckers \\n (A) tuna (B) iron (C) trash (D) garlic (E) pubs',
    'If the mass of an object gets bigger what will happen to the amount of matter contained within it? \\n (A) gets bigger (B) gets smaller',
]

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/select', methods=['GET', 'POST'])
def select():
    return jsonify(result={"questions": questions})

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    question = request.args.get('question')
    if debug:
        result = {
            'question': 'question',
            'knowledges': ['', 'selected_knowledge', 'other_knowledge'],
            'knowless_pred': 'knowless_pred',
            'knowful_pred': 'knowful_pred',
            'selected_knowledge': 'selected_knowledge',
        }
    else:
        result = rainier.run(question)

    return jsonify(result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=14411, threaded=True)
    # 14411 is the elevation of Mt Rainier, in feet

