# Rainier: Reinforced Knowledge Introspector

This repo hosts the code for the paper, [Rainier: Reinforced Knowledge Introspector for Commonsense Question Answering](https://arxiv.org/pdf/2210.03078.pdf)

## Setup

Create and activate the Conda environment:
```
conda env create -f environment.yml
conda activate rainier
```

Download the Rainier model from [here](https://drive.google.com/drive/folders/1GsuWpYvb4oAHxapMPizbEuWLZlpHUujG?usp=sharing) and put it at `/model/rainier-large.pth`

### Download data

Download the UQA data by going to `/data/` and running `python download_uqa.py`

Download the non-UQA data `non-uqa.zip` from [here](https://drive.google.com/drive/folders/1GsuWpYvb4oAHxapMPizbEuWLZlpHUujG?usp=sharing), unzip it, and put the 4 individual folders in `/data/`

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

If you would like to skip this training stage, you can download the trained model `rainier-large_stageI.pth` from [here](https://drive.google.com/drive/folders/1GsuWpYvb4oAHxapMPizbEuWLZlpHUujG?usp=sharing), and put it under `/model/`

First, generate silver knowledge from GPT-3.

If you would like to use our pre-generated data, you can download the `knowledge.zip` from [here](https://drive.google.com/drive/folders/1GsuWpYvb4oAHxapMPizbEuWLZlpHUujG?usp=sharing), unzip it, and put it at `/data/knowledge/`

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

Make sure to run `python extract_model_from_ckpt_stageII.py --load_from_ckpt ../runs/[path-to-best].pth` after the training.

