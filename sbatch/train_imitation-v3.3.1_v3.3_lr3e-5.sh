#!/bin/bash
#SBATCH --job-name=train_imitation-v3_uqa_no-lowercase_answer128_align-qk-qa_lr3e-5
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=h2lab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=240:00:00
#SBATCH --output="/gscratch/xlab/liujc/rainier/logs/%J.%x.out"

cat $0
echo "--------------------"

source "$CONDA_PREFIX/../../etc/profile.d/conda.sh"
conda activate rainier
cd /gscratch/xlab/liujc/rainier/rainier
python imitation_v3.py \
    --job_name train_imitation-v3_uqa_no-lowercase_answer128_align-qk-qa_lr3e-5 \
    --model_type allenai/unifiedqa-t5-large --batch_size 4 --accumulate_grad_batches 8 --lr 3e-5

