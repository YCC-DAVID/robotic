<h1 align="center">Robotic — LingBot-VLA 2.0 Stack-the-cubes Finetune & Real-Robot Serving</h1>

<p align="center">
  <a href="https://huggingface.co/YccHugAi/lingbot-vla-2-stackcube"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&message=HuggingFace&color=yellow"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green"></a>
</p>

## Overview

This repo finetunes **[LingBot-VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2)** (6B, MoE action expert) on our own cube-stacking data and adds a **real-robot serving path** so the policy can drive a physical **Unitree G1 Dex1** dual-arm robot over the network.

- **Task**: dual-arm cube stacking (`stack the cubes`)
- **Robot**: Unitree G1 Dex1 — 16-dim state/action (14 arm joints: 7 left + 7 right; 2 grippers), 3 cameras (top / left-wrist / right-wrist)
- **Data**: [LGG100/Stack-the-cubes](https://huggingface.co/datasets/LGG100/Stack-the-cubes) + [LGG100/Stack-the-cubes-v2](https://huggingface.co/datasets/LGG100/Stack-the-cubes-v2), mixed
- **Finetuned weights**: 🤗 [YccHugAi/lingbot-vla-2-stackcube](https://huggingface.co/YccHugAi/lingbot-vla-2-stackcube) (checkpoints at 5k / 10k / 15k / 20k steps)

> Model weights live on HuggingFace; this repo is **code only** (no weights/data checked in).

## What's added on top of upstream

| Path | Purpose |
|---|---|
| `configs/robot_configs/stackcubes.yaml` | Unitree G1 Dex1 raw↔unified state/action & camera mapping |
| `configs/vla/stackcubes/` | Training config + norm-stats config |
| `assets/norm_stats/stackcubes.json` | Precomputed normalization statistics |
| `experiment/stackcubes/` | SLURM sbatch (train / compute-norm), smoke test, HF upload util |
| `deploy/serve_stackcubes.sh` | **Policy server launcher** for real-robot inference |
| `deploy/client_stackcubes_example.py` | **Robot-side client example** (exact observation format) |

## Installation

```bash
conda create -n lingbotvla python=3.12 -y && conda activate lingbotvla
pip install -r requirements.txt          # + requirements-depth.txt for the depth branch
pip install -e .
```

Requires PyTorch 2.8.0. Base VLA weights (Qwen3-VL-4B + LingBot-VLA 2.0) go under `pretrained/`.
See upstream [LingBot-VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2) for the full pretrained-weights layout.

## Finetuning

**1. Data** — datasets must be in LeRobot **v3.0** format. If yours are v2.1, convert first:

```bash
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
  --repo-id=<name> --root=<parent_dir> --push-to-hub=false
```

Point `assets/training_data/stackcubes.txt` at the dataset dirs (one `name path` per line).

**2. Normalization stats**

```bash
sbatch experiment/stackcubes/compute_norm.sbatch      # SLURM
# writes assets/norm_stats/stackcubes.json
```

**3. Train** (2×A100/H100, FSDP2, Muon, `L1_fm` loss, depth + DINO-video distillation):

```bash
sbatch experiment/stackcubes/train_a100.sbatch        # or train.sbatch (H100)
```

Checkpoints land in `output/stackcubes/checkpoints/global_step_*/` — the deployable HF weights
are under each `hf_ckpt/`. Quick 3-step sanity check: `bash experiment/stackcubes/smoke.sh`.

## Real-robot deployment

The policy runs as a **websocket server** on a GPU machine; the robot connects as a client and
streams observations, receiving action chunks. Protocol is msgpack + websocket (openpi-compatible).

**On the inference server (GPU):**

```bash
MODEL_PATH=output/stackcubes/checkpoints/global_step_20000/hf_ckpt \
PORT=8006 GPU=0 bash deploy/serve_stackcubes.sh
```

**On the robot machine:**

```bash
python deploy/client_stackcubes_example.py --host <server-ip> --port 8006
```

Each new episode: `policy.reset("stackcubes")`, then per control step send an observation dict:

| Key | Type | Shape | Meaning |
|---|---|---|---|
| `observation.images.cam_left_high`   | uint8 | (H, W, 3) HWC | top camera |
| `observation.images.cam_left_wrist`  | uint8 | (H, W, 3) HWC | left-wrist camera |
| `observation.images.cam_right_wrist` | uint8 | (H, W, 3) HWC | right-wrist camera |
| `observation.state`                  | float32 | (16,) | 14 joints + 2 grippers |
| `task`                               | str | — | language instruction, e.g. `"stack the cubes"` |

`policy.infer(obs)` returns `{"action": ndarray}` — shape `(chunk, 16)` when `chunk_ret=True`,
else `(16,)`. First 14 dims are dual-arm joint targets, last 2 are the left/right grippers.

## Open-loop evaluation

```bash
python scripts/open_loop_eval.py \
  --model_path output/stackcubes/checkpoints/global_step_20000/hf_ckpt \
  --robo_name stackcubes --data_path <validation-dataset> --traj_ids 0 1 2
```

## Acknowledgement

Built on **[LingBot-VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2)** by Robbyant. Please refer to
the upstream repo and technical report for the base model, architecture, and license.
