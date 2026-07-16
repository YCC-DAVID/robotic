#!/bin/bash -l
# Version C norm stats: Stack-the-cubes EEF (100ep ONLY, end-effector space).
#SBATCH --job-name=stackcubes_eef100_norm
#SBATCH --partition=gpu_p
#SBATCH --account=gylab
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cy65664@uga.edu
#SBATCH --output=/scratch/cy65664/workDir/lingbot-vla-v2/experiment/stackcubes/logs/norm_eef100_%j.out
#SBATCH --error=/scratch/cy65664/workDir/lingbot-vla-v2/experiment/stackcubes/logs/norm_eef100_%j.out

set -uo pipefail
REPO=/scratch/cy65664/workDir/lingbot-vla-v2
cd "$REPO"
mkdir -p experiment/stackcubes/logs

module load CUDA/12.8.0 2>/dev/null || true
module load FFmpeg/7.1.2-GCCcore-14.3.0 2>/dev/null || true
source /apps/eb/Miniforge3/24.11.3-0/etc/profile.d/conda.sh
conda activate lingbotvla
echo "python: $(which python) | torchrun: $(which torchrun)"

export HF_HOME=/scratch/cy65664/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "===== $(hostname) $(date) job=$SLURM_JOB_ID ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm_stats.py ./configs/vla/stackcubes/stackcubes_eef100_norm.yaml \
  --data.data_name multi \
  --data.train_path assets/training_data/stackcubes_eef100.txt \
  --data.robot_config_root ./configs/robot_configs \
  --data.norm_path assets/norm_stats/stackcubes_eef100.json \
  --data.data_ratio_for_norm_compute 1

echo "===== norm compute exited rc=$? $(date) ====="
