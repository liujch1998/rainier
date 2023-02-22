#!/bin/bash
#SBATCH --job-name=train_imitation-v3.6_v3.5_half-half
#SBATCH --partition=learnlab
#SBATCH --nodes=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gpus-per-node=8
#SBATCH --constraint="volta32gb"
#SBATCH --time=72:00:00
#SBATCH --output="/private/home/ljc/rainier/logs/%J.%x.out"

wrapper="sbatch/train_imitation-v3.6.sh.wrapper"
cat $0
echo "--------------------"
cat $wrapper
echo "--------------------"

time=$(date +"%Y%m%d-%H%M%S")
srun --label ${wrapper} \
    ${time}.${SLURM_JOB_ID}.${SLURM_JOB_NAME} \
    no allenai/unifiedqa-t5-large 4
