#!/bin/bash
#SBATCH --job-name=train_imitation-v3.1-large
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=h2lab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=120:00:00
#SBATCH --output="/gscratch/xlab/liujc/rainier/logs/%J.%x.out"

cat $0
echo "--------------------"

source "$CONDA_PREFIX/../../etc/profile.d/conda.sh"
conda activate rainier
cd /gscratch/xlab/liujc/rainier/rainier
python imitation_v3.py --model_type allenai/unifiedqa-t5-large --batch_size 32

