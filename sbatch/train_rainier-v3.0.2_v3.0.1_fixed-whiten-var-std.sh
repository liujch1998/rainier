#!/bin/bash
#SBATCH --job-name=train_rainier-v3.0.2_v3.0.1_fixed-whiten-var-std
#SBATCH --partition=learnlab
#SBATCH --nodes=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --gpus-per-node=8
#SBATCH --constraint="volta32gb"
#SBATCH --time=72:00:00
#SBATCH --output="/private/home/ljc/rainier/logs/%J.%x.out"

cat $0
echo "--------------------"

time=$(date +"%Y%m%d-%H%M%S")
srun --label sbatch/train_rainier-v3.0.sh.wrapper \
    ${time}.${SLURM_JOB_ID}.${SLURM_JOB_NAME} \
    no