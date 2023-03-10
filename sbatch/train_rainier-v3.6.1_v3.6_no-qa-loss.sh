#!/bin/bash
#SBATCH --job-name=train_rainier-v3.6.1_v3.6_no-qa-loss
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
    no 0.0 1.0 "../runs_stageI/20230223-151106.4218514.train_imitation-v3.6.2_v3.6_no-qa-loss/model/ckp_4000.pth" 8.04789146111577 -0.32130728223585264
