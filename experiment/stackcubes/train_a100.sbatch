#!/bin/bash -l
# Post-training LingBot-VLA 2.0 on mixed Stack-the-cubes (v1 + v2) datasets.
# Submit:  GPUS=4 sbatch experiment/stackcubes/train.sbatch
#SBATCH --job-name=stackcubes_vla
#SBATCH --partition=gpu_p
#SBATCH --account=gylab
#SBATCH --gres=gpu:A100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=cy65664@uga.edu
#SBATCH --output=/scratch/cy65664/workDir/lingbot-vla-v2/experiment/stackcubes/logs/train_%j.out
#SBATCH --error=/scratch/cy65664/workDir/lingbot-vla-v2/experiment/stackcubes/logs/train_%j.out

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
export QWEN3_PATH=/scratch/cy65664/workDir/lingbot-vla-v2/pretrained/Qwen3-VL-4B-Instruct
export QWEN3VL_PATH=$QWEN3_PATH

# Number of GPUs = entries in CUDA_VISIBLE_DEVICES (set by SLURM gres), else count via nvidia-smi.
GPUS=$(nvidia-smi -L | wc -l)
GBS=$((GPUS * 1))   # micro_batch_size(1) * num_gpus * grad_accum(1)

echo "===== $(hostname) $(date) job=$SLURM_JOB_ID GPUS=$GPUS GBS=$GBS restarts=${SLURM_RESTART_COUNT:-0} ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/stackcubes/stackcubes.yaml \
  --data.norm_stats_file assets/norm_stats/stackcubes.json \
  --train.output_dir output/stackcubes \
  --train.micro_batch_size 1 \
  --train.gradient_accumulation_steps 1 \
  --train.global_batch_size "$GBS"

echo "===== training exited rc=$? $(date) ====="
