#!/bin/bash
#SBATCH --job-name=train_imitation-v3.5_v3.4.6
#SBATCH --partition=learnlab
#SBATCH --nodes=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gpus-per-node=8
#SBATCH --constraint="volta32gb"
#SBATCH --time=72:00:00
#SBATCH --output="/private/home/ljc/rainier/logs/%J.%x.out"

cat $0
echo "--------------------"

time=$(date +"%Y%m%d-%H%M%S")
srun --label sbatch/train_imitation-v3.5.sh.wrapper \
    ${time}.${SLURM_JOB_ID}.${SLURM_JOB_NAME} \
    no large 4
