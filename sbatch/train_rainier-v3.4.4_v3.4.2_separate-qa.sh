#!/bin/bash
#SBATCH --job-name=train_rainier-v3.4.4_v3.4.2_separate-qa
#SBATCH --partition=learnlab
#SBATCH --nodes=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --gpus-per-node=8
#SBATCH --constraint="volta32gb"
#SBATCH --time=72:00:00
#SBATCH --output="/private/home/ljc/rainier/logs/%J.%x.out"

wrapper="sbatch/train_rainier-v3.4.sh.wrapper"
cat $0
echo "--------------------"
cat $wrapper
echo "--------------------"

time=$(date +"%Y%m%d-%H%M%S")
srun --label ${wrapper} \
    ${time}.${SLURM_JOB_ID}.${SLURM_JOB_NAME} \
    no "../runs_stageI/20230221-100920.4031019.train_imitation-v3.6_v3.5_half-half/model/ckp_5000.pth" 12.851015507843925 0.02181289785491071