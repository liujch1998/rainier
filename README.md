# Rainier: Reinforced Knowledge Introspector

This repo hosts the code for the paper, [Rainier: Reinforced Knowledge Introspector for Commonsense Question Answering](https://arxiv.org/pdf/2210.03078.pdf)

## Setup

Create and activate the Conda environment:
```
conda env create -f environment.yml
conda activate rainier
```

Download the Rainier model from [here](https://drive.google.com/drive/folders/1GsuWpYvb4oAHxapMPizbEuWLZlpHUujG?usp=sharing) and put it at `/model/rainier-large.pth`

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
--eval_baseline				Additionally evaluate the no-knowledge baseline.
--ckpt [path-to-rainier]    The path to Rainier model ckpt. The default value is ../model/rainier-large.pth
--load_from_ckpt [path]     This loads the Rainier model ckpt from a raw training ckpt file, and overrides the --ckpt parameter.
```

## Training the Rainier model

The Rainier model is trained in two stages.

### Stage I: Imitation Learning

TODO

### Stage II: Reinforcement Learning

We trained this stage using 8x RTX6000 GPUs, each has 24G memory.

To train Rainier with the default setting, go to the `/rainier/` directory and run
```
python main.py --mode train
```
This will train Rainier on all seen datasets, with UnifiedQA-large as the QA model.
You can track the training in Tensorboard, and view the (dev set) output knowledge in `[path-to-save-dir]/knowledge` and the inference results in `[path-to-save-dir]/inference/`.

Some flags you can set (see the full list in `args.py`):
```
--train_tasks [task-list]   Please choose a subset from the full list (which is also the default value): obqa,arc_e,arc_h,ai2sci_e,ai2sci_m,csqa,qasc,piqa,siqa,wg. Write your choice as a comma-separated list.
--eval_baseline				Additionally evaluate the no-knowledge baseline.
--load_from_ckpt [path]     This resumes training from an existing ckpt
```

