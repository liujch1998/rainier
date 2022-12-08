#!/bin/bash
#SBATCH --job-name=eval_imitation-v3-large_imitation-v3-large
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

source "$CONDA_PREFIX/../../etc/profile.d/conda.sh"
conda activate rainier_new_transformers
cd /gscratch/xlab/liujc/rainier/rainier
python main.py --mode eval \
    --eval_tasks obqa,arc_e,arc_h,ai2sci_e,ai2sci_m,csqa,qasc,piqa,siqa,wg \
    --load_from_ckpt ../runs_stageI/Nov21_20-02-36_learnfair0794/model/ckp_50000.pth

