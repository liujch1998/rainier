#!/bin/bash
#SBATCH --job-name=train_rainier-3b_stageI_new
#SBATCH --partition=devlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=8
#SBATCH --constraint="volta32gb"
#SBATCH --time=72:00:00
#SBATCH --output="/private/home/ljc/rainier/logs/%J.%x.out"

cat $0
echo "--------------------"

module load anaconda3
source "/public/apps/anaconda3/2022.05/etc/profile.d/conda.sh"
conda activate rainier
cd /private/home/ljc/rainier/rainier
python imitation_new.py \
    --model_type t5-3b \
    --batch_size 32 --accumulate_grad_batches 2 --total_steps 100000

