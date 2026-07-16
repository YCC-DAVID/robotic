#!/bin/bash
# ==============================================================================
# LingBot-VLA 2.0 — Stack-the-cubes 推理服务端启动脚本
#
# 在**推理服务器**（和机器人同一局域网的那台带 GPU 的机器）上运行。
# 起一个 websocket policy server，机器人端用 deploy/client_stackcubes_example.py
# 里的 WebsocketClientPolicy 连过来推理。
#
# 用法:
#   # 最简单，用默认（GPU 0, 端口 8006, joint 基线 checkpoint + robo_name=stackcubes）
#   bash deploy/serve_stackcubes.sh
#
#   # EEF 模型（配 g1-client openpi/main_eef.py 时用这个！模型和 robo_name 必须成对）
#   MODEL_PATH=output/stackcubes_eef/checkpoints/global_step_20000/hf_ckpt \
#   ROBO_NAME=stackcubes_eef PORT=8000 GPU=0 bash deploy/serve_stackcubes.sh
#
# 通信协议与 openpi 一致（msgpack + websocket + /healthz）。对 openpi 无状态
# client（如 main_eef.py，不发 reset 握手、语言用 "prompt" key、读 result["actions"]）
# 已做兼容：server 会用 ROBO_NAME 在首个请求时自动 reset，prompt->task 自动映射，
# 返回里同时带 "action" 和 "actions"。
# ⚠️ MODEL_PATH 和 ROBO_NAME 必须匹配（joint 模型配 stackcubes，EEF 模型配
#    stackcubes_eef / stackcubes_eef100）——配错 = 归一化/动作空间错乱 = 机器人抽搐。
# ==============================================================================
set -euo pipefail

# 切到仓库根目录（本脚本在 deploy/ 下）
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# ---- 可配置项（都能用环境变量覆盖）----
MODEL_PATH="${MODEL_PATH:-output/stackcubes/checkpoints/global_step_20000/hf_ckpt}"
ROBO_NAME="${ROBO_NAME:-stackcubes}"   # 必须和 MODEL_PATH 的动作空间匹配!
PORT="${PORT:-8006}"
GPU="${GPU:-0}"
USE_LENGTH="${USE_LENGTH:-50}"      # 每次 infer 消费的 action chunk 长度
CHUNK_RET="${CHUNK_RET:-true}"      # true=一次返回整段 chunk；false=每次 infer 返回一步
USE_COMPILE="${USE_COMPILE:-true}"  # torch.compile：首次预热慢，之后快很多
SMOOTH="${SMOOTH:-savgol}"          # 服务端 chunk 零相位平滑: savgol|ema|none
                                    # (实测模型原始输出高频抖动 = 17.7x 示教, savgol 建议开)
SMOOTH_WINDOW="${SMOOTH_WINDOW:-7}" # savgol 窗口(奇数, 30fps 下 7 步 ≈ 0.23s)

# Qwen3-VL 基座路径（tokenizer / vision）。默认指向本仓库 pretrained/ 下的权重。
export QWEN3VL_PATH="${QWEN3VL_PATH:-$REPO/pretrained/Qwen3-VL-4B-Instruct}"
export QWEN3_PATH="${QWEN3_PATH:-$QWEN3VL_PATH}"

# 全离线，别去连 HF（权重都在本地）
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU}"
# 限制 CPU 线程数：推理主体在 GPU，CPU 只做轻量预处理；不限的话 torch 默认开满
# 全核，模型构造/加载期间的全核 AVX 负载足以让工作站热节流(CPU 降频)。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

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
echo "   ROBO_NAME   = $ROBO_NAME   (首个请求自动 reset 用；必须匹配 MODEL_PATH 的动作空间)"
echo "==================================================================="

# 粗略的模型/robo_name 匹配检查：EEF 模型目录名含 eef，robo_name 也应含 eef
case "$MODEL_PATH" in
  *eef*) case "$ROBO_NAME" in *eef*) ;; *) echo "[WARN] MODEL_PATH 像 EEF 模型但 ROBO_NAME=$ROBO_NAME 不含 eef —— 配错会导致机器人抽搐!" >&2 ;; esac ;;
  *)     case "$ROBO_NAME" in *eef*) echo "[WARN] ROBO_NAME=$ROBO_NAME 是 EEF 配置但 MODEL_PATH 像 joint 模型 —— 配错会导致机器人抽搐!" >&2 ;; esac ;;
esac

exec python -m deploy.lingbot_vla_v2_policy \
  --model_path "$MODEL_PATH" \
  --port "$PORT" \
  --use_length "$USE_LENGTH" \
  --chunk_ret "$CHUNK_RET" \
  --use_compile "$USE_COMPILE" \
  --robo_name "$ROBO_NAME" \
  --smooth "$SMOOTH" \
  --smooth_window "$SMOOTH_WINDOW"
