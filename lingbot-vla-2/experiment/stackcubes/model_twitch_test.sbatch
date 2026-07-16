#!/bin/bash -l
# Offline model twitch test: feed recorded EEF dataset frames to the deployment
# policy and measure EEF/joint jitter. Isolates model output from IK + stitching.
# Submit:  sbatch experiment/stackcubes/model_twitch_test.sbatch [EPISODE] [STRIDE]
#SBATCH --job-name=twitch_test
#SBATCH --partition=gpu_p
#SBATCH --account=gylab
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/cy65664/workDir/lingbot-vla-v2/experiment/stackcubes/logs/twitch_%j.out
#SBATCH --error=/scratch/cy65664/workDir/lingbot-vla-v2/experiment/stackcubes/logs/twitch_%j.out

set -uo pipefail
REPO=/scratch/cy65664/workDir/lingbot-vla-v2
cd "$REPO"; mkdir -p experiment/stackcubes/logs

module load CUDA/12.8.0 2>/dev/null || true
module load FFmpeg/7.1.2-GCCcore-14.3.0 2>/dev/null || true
source /apps/eb/Miniforge3/24.11.3-0/etc/profile.d/conda.sh
conda activate lingbotvla

export HF_HOME=/scratch/cy65664/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export QWEN3VL_PATH=$REPO/pretrained/Qwen3-VL-4B-Instruct
export QWEN3_PATH=$QWEN3VL_PATH
# PyAV(FFmpeg) + torch 的 OpenMP 运行时会打架，曾把 load_state_dict 卡死在 futex 上；
# 限线程 + 无缓冲输出
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

EP=${1:-0}; STRIDE=${2:-25}; SMOOTH=${3:-none}; VER=${4:-stackcubes_eef}
echo "===== $(hostname) $(date) job=$SLURM_JOB_ID  episode=$EP stride=$STRIDE smooth=$SMOOTH ver=$VER ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Lustre 间歇性拥堵会把权重随机读拖到 ~1MB/s（诊断 47059913: 单 shard 25min）。
# 先把 checkpoint 顺序预拷到节点本地盘（顺序读对 Lustre 友好得多），再从本地加载。
# 目录结构要保持 <root>/lingbotvla_cli.yaml + <root>/checkpoints/.../hf_ckpt（server 按相对层级找 cli yaml）。
LOCAL=${TMPDIR:-/tmp}/${VER}_ckpt
SRC=output/$VER
echo "[prestage] $(date +%T) copying checkpoint to $LOCAL ..."
mkdir -p "$LOCAL/checkpoints/global_step_20000"
cp -v "$SRC/lingbotvla_cli.yaml" "$LOCAL/" || exit 1
time cp -r "$SRC/checkpoints/global_step_20000/hf_ckpt" "$LOCAL/checkpoints/global_step_20000/" || exit 1
echo "[prestage] $(date +%T) done: $(du -sh $LOCAL | cut -f1)"

python -u experiment/stackcubes/model_twitch_test.py \
  --model_path "$LOCAL/checkpoints/global_step_20000/hf_ckpt" \
  --robo_name "$VER" \
  --episode "$EP" --stride "$STRIDE" --smooth "$SMOOTH"

echo "===== twitch test exited rc=$? $(date) ====="
