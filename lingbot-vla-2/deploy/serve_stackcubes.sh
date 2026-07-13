#!/bin/bash
# ==============================================================================
# LingBot-VLA 2.0 — Stack-the-cubes 推理服务端启动脚本
#
# 在**推理服务器**（和机器人同一局域网的那台带 GPU 的机器）上运行。
# 起一个 websocket policy server，机器人端用 deploy/client_stackcubes_example.py
# 里的 WebsocketClientPolicy 连过来推理。
#
# 用法:
#   # 最简单，用默认（GPU 0, 端口 8006, 最终 checkpoint）
#   bash deploy/serve_stackcubes.sh
#
#   # 指定 checkpoint / 端口 / GPU
#   MODEL_PATH=output/stackcubes/checkpoints/global_step_20000/hf_ckpt \
#   PORT=8006 GPU=0 bash deploy/serve_stackcubes.sh
#
# 通信协议与 openpi 一致（msgpack + websocket + /healthz），机器人端可复用
# openpi-client 或本仓库 deploy/websocket_client_policy.py。
# ==============================================================================
set -euo pipefail

# 切到仓库根目录（本脚本在 deploy/ 下）
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# ---- 可配置项（都能用环境变量覆盖）----
MODEL_PATH="${MODEL_PATH:-output/stackcubes/checkpoints/global_step_20000/hf_ckpt}"
PORT="${PORT:-8006}"
GPU="${GPU:-0}"
USE_LENGTH="${USE_LENGTH:-50}"      # 每次 infer 消费的 action chunk 长度
CHUNK_RET="${CHUNK_RET:-true}"      # true=一次返回整段 chunk；false=每次 infer 返回一步
USE_COMPILE="${USE_COMPILE:-true}"  # torch.compile：首次预热慢，之后快很多

# Qwen3-VL 基座路径（tokenizer / vision）。默认指向本仓库 pretrained/ 下的权重。
export QWEN3VL_PATH="${QWEN3VL_PATH:-$REPO/pretrained/Qwen3-VL-4B-Instruct}"
export QWEN3_PATH="${QWEN3_PATH:-$QWEN3VL_PATH}"

# 全离线，别去连 HF（权重都在本地）
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU}"

if [ ! -d "$MODEL_PATH" ]; then
  echo "[ERROR] MODEL_PATH 不存在: $MODEL_PATH" >&2
  echo "        用 MODEL_PATH=<.../global_step_XXXXX/hf_ckpt> 指定一个 checkpoint" >&2
  exit 1
fi
if [ ! -f "$MODEL_PATH/../../../lingbotvla_cli.yaml" ]; then
  echo "[WARN] 未在 $MODEL_PATH 上三级找到 lingbotvla_cli.yaml；" >&2
  echo "       server 需要它来还原训练配置（robot config / norm stats）。" >&2
fi

echo "==================================================================="
echo " LingBot-VLA 2.0  Stack-the-cubes  policy server"
echo "   MODEL_PATH  = $MODEL_PATH"
echo "   PORT        = $PORT   (ws://<this-host>:$PORT)"
echo "   GPU         = $GPU"
echo "   USE_LENGTH  = $USE_LENGTH   CHUNK_RET=$CHUNK_RET   USE_COMPILE=$USE_COMPILE"
echo "   QWEN3VL     = $QWEN3VL_PATH"
echo "   robo_name   = stackcubes   (客户端 reset 时传这个)"
echo "==================================================================="

exec python -m deploy.lingbot_vla_v2_policy \
  --model_path "$MODEL_PATH" \
  --port "$PORT" \
  --use_length "$USE_LENGTH" \
  --chunk_ret "$CHUNK_RET" \
  --use_compile "$USE_COMPILE"
