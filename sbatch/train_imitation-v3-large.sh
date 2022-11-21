#!/bin/bash
#SBATCH --job-name=train_imitation-v3-large
#SBATCH --partition=devlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=4
#SBATCH --constraint="volta32gb"
#SBATCH --time=72:00:00
#SBATCH --output="/private/home/ljc/rainier/logs/%J.%x.out"

cat $0
echo "--------------------"

module load anaconda3
source "/public/apps/anaconda3/2022.05/etc/profile.d/conda.sh"
conda activate rainier
cd /private/home/ljc/rainier/rainier
python imitation_v3.py --model_type allenai/unifiedqa-v2-t5-large-1251000 --batch_size 32

