# Rainier: Reinforced Knowledge Introspector

This repo hosts the code for the paper, [Rainier: Reinforced Knowledge Introspector for Commonsense Question Answering](https://arxiv.org/pdf/2210.03078.pdf)

## Resources

**Model**: Our Rainier model is now on huggingface model hub!
```
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-large')
model = T5ForConditionalGeneration.from_pretrained('liujch1998/rainier-large')

question = "You can share files with someone if you have a connection to a what? \n (A) freeway (B) radio (C) wires (D) computer network (E) electrical circuit"
input_ids = tokenizer(question, return_tensors='pt').input_ids
output_ids = model.generate(input_ids, do_sample=True, top_p=0.5)
knowledge = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
knowledge
```
```
Files can be shared over a computer network.
```

**Knowledge**: We release the commonsense datasets augmented with Rainier-generated knowledge.
You can download the `knowledge_rainier.json` file from [our Google Drive folder](https://drive.google.com/drive/folders/1GsuWpYvb4oAHxapMPizbEuWLZlpHUujG?usp=sharing).

## Setup

Create and activate the Conda environment:
```
conda env create -f environment.yml
conda activate rainier
```

Install [gsutil](https://cloud.google.com/storage/docs/gsutil_install).

### Download model

**Download the Rainier model**: Go to `/model/` and run `gdown 1qmxFTENNITA16_54dkqR6pHMDofa3Jee`
Alternatively, you can download the `rainier-large.pth` file from [our Google Drive folder](https://drive.google.com/drive/folders/1GsuWpYvb4oAHxapMPizbEuWLZlpHUujG?usp=sharing) and put it under `/model/`

### Download data

**Download the UQA data**: Go to `/data/` and run `python download_uqa.py`

**Download the non-UQA data**: Go to `/data/` and run `gdown 1vfJQnqeRzr9MXPQmtbrAsQUuWZD1bZqF`
Alternatively, you can download the `non-uqa.zip` file from [our Google Drive folder](https://drive.google.com/drive/folders/1GsuWpYvb4oAHxapMPizbEuWLZlpHUujG?usp=sharing), put it under `/data/` and unzip it. Make sure the 4 individual folders are directly under `/data/`

## Running inference

Running inference requires a GPU with at least 22G memory.
If that doesn't fit your memory, consider parallelizing on multiple GPUs, or using a smaller `--batch_size`.

To run inference with the default setting, go to the `/rainier/` directory and run
```
python main.py --mode eval
```
This will evaluate the dev split of all seen and unseen datasets, with Rainier-large as the knowledge introspector and UnifiedQA-large as the QA model.
You can view the output knowledge in `/model/knowledge/` and the inference results in `/model/inference/`.

Some flags you can set (see the full list in `args.py`):
```
--eval_split [dev|test]     The dataset split you want to evaluate. Some test data does not have gold labels so we provide utility scripts to convert the inference results to leaderboard submission files.
--eval_tasks [task-list]    Please choose a subset from the full list (which is also the default value): obqa,arc_e,arc_h,ai2sci_e,ai2sci_m,csqa,qasc,piqa,siqa,wg,numersense,riddlesense,quartz,hellaswag. Write your choice as a comma-separated list.
--eval_baseline             Additionally evaluate the no-knowledge baseline.
--eval_ckpt [path]          The path to Rainier model ckpt. The default value is ../model/rainier-large.pth
--load_from_ckpt [path]     This loads the Rainier model ckpt from a raw training ckpt file, and overrides the --ckpt parameter.
```

## Training the Rainier model

The Rainier model is trained in two stages.

### Stage I: Imitation Learning

We trained this stage using 1x RTX6000 GPU with 24G memory.

If you would like to skip this training stage, you can download a copy of our ckpt.
Go to `/model/` and run `gdown 1PeL3E7UreVIHKOkLNSyzgyAYoab-MA5N`
Alternatively, you can download the `rainier-large_stageI.pth` file from [our Google Drive folder](https://drive.google.com/drive/folders/1GsuWpYvb4oAHxapMPizbEuWLZlpHUujG?usp=sharing) and put it under `/model/`

First, generate silver knowledge from GPT-3.

If you would like to use our pre-generated data, you can download a copy of our pre-generated knowledge.
Go to `/data/` and run `gdown 1V6Za8BfEwWa4xRgXcVEFhS8tWepHZPAw`
Alternatively, you can download the `knowledge_gkp.zip` file from [our Google Drive folder](https://drive.google.com/drive/folders/1GsuWpYvb4oAHxapMPizbEuWLZlpHUujG?usp=sharing), unzip it and put it under `/data/`

Otherwise, you can generate the knowledge yourself by going to the `/rainier/` directory and run
```
sh generate_knowledge_gkp.sh
```
Remember to set the `OPENAI_API_KEY` envvar beforehand, and be ready to spend a lot of money ;)

Then, you can start Stage I training by going to the `/rainier/` directory and run
```
python imitation.py 
```
This will train on all seen datasets, using silver knowledge as supervision.
You can track the training in Tensorboard.
The best model ckpt will be saved under `/runs/imitation/`.
Make sure to run `python extract_model_from_ckpt_stageI.py ../runs/imitation/[path-to-best].ckpt` before proceeding to the next stage.
This extracts the model state dict and puts it at `/model/rainier-large_stageI.pth`

### Stage II: Reinforcement Learning

We trained this stage using 8x RTX6000 GPUs, each has 24G memory.

To train Stage II with the default setting, go to the `/rainier/` directory and run
```
python main.py --mode train
```
This will train Rainier on all seen datasets, with UnifiedQA-large as the QA model.
You can track the training in Tensorboard, and view the (dev set) output knowledge in `/runs/[path-to-save-dir]/knowledge/` and the inference results in `/runs/[path-to-save-dir]/inference/`.

Some flags you can set (see the full list in `args.py`):
```
--train_tasks [task-list]   Please choose a subset from the full list (which is also the default value): obqa,arc_e,arc_h,ai2sci_e,ai2sci_m,csqa,qasc,piqa,siqa,wg. Write your choice as a comma-separated list.
--eval_baseline             Additionally evaluate the no-knowledge baseline.
--model_ckpt [path]         The path to stage I model ckpt. The default value is ../model/rainier-large_stageI.pth
--load_from_ckpt [path]     This resumes training from an existing ckpt.
```

Make sure to run `python extract_model_from_ckpt_stageII.py --load_from_ckpt ../runs/[path-to-best].pth` after the training, so that you can use the trained Rainier model for inference.

