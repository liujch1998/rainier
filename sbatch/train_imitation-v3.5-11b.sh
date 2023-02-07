#!/bin/bash
#SBATCH --job-name=train_imitation-v3.5-11b
#SBATCH --partition=learnlab
#SBATCH --nodes=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --gpus-per-node=8
#SBATCH --constraint="volta32gb"
#SBATCH --time=72:00:00
#SBATCH --output="/private/home/ljc/rainier/logs/%J.%x.out"

cat $0
echo "--------------------"

srun --label sbatch/train_imitation-v3.5-11b.sh.wrapper
