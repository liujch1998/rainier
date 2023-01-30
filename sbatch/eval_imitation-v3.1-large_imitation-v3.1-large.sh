#!/bin/bash
#SBATCH --job-name=eval_imitation-v3.1-large_imitation-v3.1-large
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

cd /gscratch/xlab/liujc/rainier/rainier
source "$CONDA_PREFIX/../../etc/profile.d/conda.sh"
conda activate rainier2
export LD_LIBRARY_PATH=/gscratch/xlab/liujc/anaconda3/envs/rainier2/lib:/lib64:$LD_LIBRARY_PATH
python main.py --mode eval \
    --eval_tasks obqa,arc_e,arc_h,ai2sci_e,ai2sci_m,csqa,qasc,piqa,siqa,wg \
    --load_from_ckpt ../runs_stageI/Dec08_05-15-21_g3022/model/ckp_50000.pth \
    --eval_baseline