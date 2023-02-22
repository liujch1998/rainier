#!/bin/bash
#SBATCH --job-name=train_imitation-v3_uqa_no-lowercase_answer128_align-qk-qa_qka-loss_accelerate
#SBATCH --partition=learnlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gpus-per-node=8
#SBATCH --constraint="volta32gb"
#SBATCH --time=72:00:00
#SBATCH --output="/private/home/ljc/rainier/logs/%J.%x.out"

cat $0
echo "--------------------"

cd /private/home/ljc/rainier/rainier
module load anaconda3
source "/public/apps/anaconda3/2022.05/etc/profile.d/conda.sh"
conda activate rainier2
export LD_LIBRARY_PATH=~/.conda/envs/rainier2/lib:$LD_LIBRARY_PATH

accelerate launch --config_file ../accelerate.cfg imitation_v3.py \
    --job_name train_imitation-v3_uqa_no-lowercase_answer128_align-qk-qa_qka-loss_accelerate \
    --model_type allenai/unifiedqa-t5-large --batch_size 4 --qka_loss
