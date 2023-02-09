#!/bin/bash
#SBATCH --job-name=train_imitation-v3_uqa_no-lowercase_answer128_align-qk-qa_qka-loss_accelerate_gather_node2
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

srun --label train_imitation-v3_uqa_no-lowercase_answer128_align-qk-qa_qka-loss_accelerate_gather_node2.sh.wrapper
