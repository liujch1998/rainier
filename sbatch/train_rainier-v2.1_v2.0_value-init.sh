#!/bin/bash
#SBATCH --job-name=train_rainier-large_uqa-large_value-init
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=h2lab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:8
#SBATCH --time=480:00:00
#SBATCH --output="/gscratch/xlab/liujc/sbatch/logs/%J.%x.out"

cat $0
echo "--------------------"

source "$CONDA_PREFIX/../../etc/profile.d/conda.sh"
conda activate rainier
cd /gscratch/xlab/liujc/rainier/rainier
python main.py --mode train \
    --gain 3.575475037847048 --bias 0.032954977862281395 \
    --use_model_ckpt_for_value

