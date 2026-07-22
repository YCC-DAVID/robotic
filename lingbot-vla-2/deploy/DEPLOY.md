# Stack-the-cubes real-robot deployment

Two machines are involved, both separate from the training cluster:

```
推理服务器 (GPU, 这份指南)                      转发机 (机器人旁, 无需GPU)
  git clone this repo                              g1-client repo
  download weights from HF                         python openpi/main_eef.py ...
  bash deploy/serve_stackcubes.sh  <--websocket-->  (openpi wire protocol)
```

This file covers the **server side**. The client side is `g1-client`
(`openpi/main_eef.py` + `eef_kinematics.py`), a separate repo.

## 1. What you need on the inference server

| Category | Source | Notes |
|---|---|---|
| Code | this repo (already have it if reading this) | `deploy/`, `configs/robot_configs/`, `assets/norm_stats/`, `deploy/serving_configs/` |
| Model weights | HF `YccHugAi/lingbot-vla-2-stackcube-ablations` | pick a subfolder + step — see "Which checkpoint" below |
| Qwen3-VL base | HF `Qwen/Qwen3-VL-4B-Instruct` | tokenizer + vision tower, required at inference |
| Python env | `requirements.txt` | `requirements-depth.txt` is **training-only** (MoGe/MoRGBD/DINO teachers) — inference never loads them, don't bother installing |

### Which checkpoint to use

| HF subfolder | Data | Action space | Training exposure | Recommendation |
|---|---|---|---|---|
| `joint100/` | 100ep | joint angles | 0.42 epoch | baseline |
| `eef/` | 150ep (v1+v2, mixed prompt) | EEF pose | 0.29 epoch | has a known prompt-mismatch issue in ~1/3 of the data |
| `eef100/` | 100ep, clean prompt | EEF pose | 0.42 epoch | better than `eef/` in every offline metric |
| **`eef100_10ep/`** | 100ep, clean prompt | EEF pose | **10 full epochs** | **best available — start here** |

Within `eef100_10ep/`, use the **final step (`global_step_59979`)** unless you have
a specific reason to compare intermediate checkpoints (10k/20k/30k/40k/50k are
also uploaded).

```bash
pip install huggingface_hub
huggingface-cli download YccHugAi/lingbot-vla-2-stackcube-ablations \
  --include "eef100_10ep/global_step_59979/*" \
  --local-dir /tmp/hf_download
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir /path/to/Qwen3-VL-4B-Instruct
```

## 2. Directory layout the server expects

The server finds its training config by walking **up three levels from
`MODEL_PATH`** (`<...>/lingbotvla_cli.yaml`), so the checkpoint must sit at:

```
output/stackcubes_eef100/lingbotvla_cli.yaml                       <- copy from deploy/serving_configs/stackcubes_eef100/
output/stackcubes_eef100/checkpoints/global_step_59979/hf_ckpt/    <- the HF download above
```

```bash
mkdir -p output/stackcubes_eef100/checkpoints/global_step_59979
cp deploy/serving_configs/stackcubes_eef100/lingbotvla_cli.yaml output/stackcubes_eef100/
mv /tmp/hf_download/eef100_10ep/global_step_59979 output/stackcubes_eef100/checkpoints/global_step_59979/hf_ckpt
```

## 3. Launch

```bash
MODEL_PATH=output/stackcubes_eef100/checkpoints/global_step_59979/hf_ckpt \
ROBO_NAME=stackcubes_eef100 \
PORT=8000 \
GPU=0 \
QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct \
bash deploy/serve_stackcubes.sh
```

| Env var | Default | What it does |
|---|---|---|
| `MODEL_PATH` | `output/stackcubes/checkpoints/global_step_20000/hf_ckpt` | the hf_ckpt directory |
| `ROBO_NAME` | `stackcubes` | robot config name; **must match** MODEL_PATH's action space (joint vs EEF) or normalization is silently wrong — the script WARNs loudly on an obvious mismatch, but double-check |
| `PORT` | 8006 | websocket port; `main_eef.py`'s default is `8000` — pass one to match the other |
| `SMOOTH` | `savgol` | server-side zero-phase smoothing of each action chunk; measured to cut within-chunk jitter ~4-5x with no loss of tracking accuracy. Leave on. |
| `USE_COMPILE` | `true` | first inference is slow (compiling); subsequent ones are fast |

Health check: `curl http://<server-ip>:<port>/healthz` → `OK`.

Startup should take **~1-2 minutes** (weights load straight to GPU). If it
takes much longer, something regressed — the old CPU-staging load path could
take 20+ minutes and was known to make a workstation's CPU thermal-throttle;
this was fixed and the fix is in this checkout.

## 4. Client-side launch (on the machine next to the robot, g1-client repo)

```bash
python openpi/main_eef.py \
  --iface <network interface to the robot> \
  --server-host <this server's IP> \
  --server-port 8000 \
  --control-hz 30 \
  --exec-steps 25 \
  --blend-steps 10 \
  --velocity-limit 3 \
  --prompt "Stack the blocks by color: put the red block in the center, then stack the blue block on the red block, then stack the yellow block on the blue block."
```

Every flag here overrides a default that is wrong or too conservative for
this checkpoint:

| Flag | Why it must be explicit |
|---|---|
| `--control-hz 30` | client defaults to 15Hz; the data is 30fps — leaving this at default runs the model at half speed with misaligned state feedback |
| `--exec-steps 25` | defaults to 0 (execute the full 50-step chunk); flow-matching chunks drift more in their back half, so replanning after 25 steps instead of 50 keeps the model closer to its own recent observations |
| `--blend-steps 10` | cross-fades a chunk swap instead of snapping; raised from the default (5) because measured chunk-to-chunk seam jumps (23-59mm) are the largest remaining source of visible jitter after server-side smoothing |
| `--velocity-limit 3` | defaults to 8 rad/s (barely a limit); this is a safety floor so any residual jitter executes as a slow correction, not a snap |
| `--prompt` | defaults to `"pick the red bottle"`, which the model has never seen — always pass the exact training sentence above |

`main_eef.py` is an **openpi-protocol client** (stateless, sends `prompt` not
`task`, expects `result["actions"]`) — this server is compatible with it
out of the box (auto-resets on first request via `ROBO_NAME`, aliases the
response key, decodes JPEG bytes if `--send-jpeg` is used).

## 5. What NOT to bring to the inference server

- `pretrained/moge-2-*`, `pretrained/*/depth/`, `pretrained/*/dino_video/` — training-only distillation teachers, never loaded at inference (confirmed: they add ~3.4-7% to training step time and zero code path touches them in `deploy/`).
- `requirements-depth.txt` and its dependencies.
- The base `lingbot-vla-v2-6b` pretrained checkpoint — inference loads weights straight from the finetuned `hf_ckpt`, not the pretrain checkpoint.
