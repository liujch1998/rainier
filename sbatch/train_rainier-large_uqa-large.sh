#!/bin/bash
#SBATCH --job-name=train_rainier-large_uqa-large
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=xlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:8
#SBATCH --time=480:00:00
#SBATCH --output="/gscratch/xlab/liujc/rainier/logs/%J.%x.out"

cat $0
echo "--------------------"

source "$CONDA_PREFIX/../../etc/profile.d/conda.sh"
conda activate rainier
cd /gscratch/xlab/liujc/rainier/rainier
python main.py --mode train --eval_baseline

