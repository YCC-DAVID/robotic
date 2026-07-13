#!/bin/bash
set -x
cd /scratch/cy65664/workDir/lingbot-vla-v2
module load CUDA/12.8.0 2>/dev/null || true
module load FFmpeg/7.1.2-GCCcore-14.3.0 2>/dev/null || true
source /apps/eb/Miniforge3/24.11.3-0/etc/profile.d/conda.sh
conda activate lingbotvla
export HF_HOME=/scratch/cy65664/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export QWEN3_PATH=/scratch/cy65664/workDir/lingbot-vla-v2/pretrained/Qwen3-VL-4B-Instruct
export QWEN3VL_PATH=$QWEN3_PATH
torchrun --nnodes=1 --nproc-per-node 1 --master-port=62511 \
  tasks/vla/train_lingbotvla.py ./configs/vla/stackcubes/stackcubes.yaml \
  --data.norm_stats_file assets/norm_stats/stackcubes.json \
  --train.output_dir output/stackcubes_smoke \
  --train.micro_batch_size 1 --train.gradient_accumulation_steps 1 --train.global_batch_size 1 \
  --train.max_steps 3 --train.save_steps 100000 --train.use_compile false
echo "SMOKE_EXIT=$?"
