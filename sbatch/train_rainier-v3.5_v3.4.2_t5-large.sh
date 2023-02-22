#!/bin/bash
#SBATCH --job-name=train_rainier-v3.5_v3.4.2_t5-large
#SBATCH --partition=learnlab
#SBATCH --nodes=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --gpus-per-node=8
#SBATCH --constraint="volta32gb"
#SBATCH --time=72:00:00
#SBATCH --output="/private/home/ljc/rainier/logs/%J.%x.out"

wrapper="sbatch/train_rainier-v3.5.sh.wrapper"
cat $0
echo "--------------------"
cat $wrapper
echo "--------------------"

time=$(date +"%Y%m%d-%H%M%S")
srun --label ${wrapper} \
    ${time}.${SLURM_JOB_ID}.${SLURM_JOB_NAME} \
    no "../runs_stageI/20230221-101020.4031090.train_imitation-v3.6.1_v3.6_t5-large/model/ckp_10000.pth"
