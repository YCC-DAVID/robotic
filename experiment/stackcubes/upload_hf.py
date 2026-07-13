#!/usr/bin/env python
"""Serially upload all Stack-the-cubes finetune checkpoints (deployable hf_ckpt only)
to the HF repo, then write a README. Waits for step_20000 to finish training first.
"""
import os
import time
from pathlib import Path
from huggingface_hub import HfApi

REPO = "YccHugAi/lingbot-vla-2-stackcube"
CKPT_ROOT = Path("/scratch/cy65664/workDir/lingbot-vla-v2/output/stackcubes/checkpoints")
STEPS = [5000, 10000, 15000, 20000]

api = HfApi()


def hf_ckpt_ready(step: int) -> bool:
    d = CKPT_ROOT / f"global_step_{step}" / "hf_ckpt"
    if not d.is_dir():
        return False
    idx = d / "model.safetensors.index.json"
    shards = list(d.glob("model-*-of-*.safetensors"))
    if not idx.exists() or len(shards) < 6:
        return False
    # size-stable check: all shards unchanged for 30s
    sizes = {s: s.stat().st_size for s in shards}
    time.sleep(30)
    for s, sz in sizes.items():
        if not s.exists() or s.stat().st_size != sz:
            return False
    return True


def uploaded_already(step: int) -> bool:
    try:
        files = api.list_repo_files(REPO)
    except Exception:
        return False
    return any(f.startswith(f"global_step_{step}/model.safetensors.index.json") for f in files)


def upload(step: int):
    src = CKPT_ROOT / f"global_step_{step}" / "hf_ckpt"
    print(f"[{time.strftime('%H:%M:%S')}] >>> uploading global_step_{step} from {src}", flush=True)
    api.upload_folder(
        repo_id=REPO,
        folder_path=str(src),
        path_in_repo=f"global_step_{step}",
        commit_message=f"Add checkpoint global_step_{step} (deployable hf_ckpt)",
    )
    print(f"[{time.strftime('%H:%M:%S')}] <<< DONE global_step_{step}", flush=True)


for step in STEPS:
    if uploaded_already(step):
        print(f"[{time.strftime('%H:%M:%S')}] skip global_step_{step} (already on hub)", flush=True)
        continue
    # wait until this checkpoint's hf_ckpt is fully written
    waited = 0
    while not hf_ckpt_ready(step):
        if waited % 300 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] waiting for global_step_{step} hf_ckpt ...", flush=True)
        time.sleep(30)
        waited += 30
    upload(step)

# README
readme = """---
license: apache-2.0
library_name: transformers
tags:
- robotics
- vla
- lingbot-vla
- lerobot
- manipulation
---

# LingBot-VLA 2.0 — Stack-the-cubes finetune

Finetune of **LingBot-VLA 2.0** (6B, MoE action expert) on the mixed
[LGG100/Stack-the-cubes](https://huggingface.co/datasets/LGG100/Stack-the-cubes) +
[LGG100/Stack-the-cubes-v2](https://huggingface.co/datasets/LGG100/Stack-the-cubes-v2)
datasets (Unitree G1 Dex1, dual-arm cube stacking).

## Checkpoints

Each `global_step_*/` folder holds the deployable HF-format weights
(`model-0000x-of-00006.safetensors` + tokenizer/config), loadable directly.

| Folder | Step |
|---|---|
| `global_step_5000/`  | 5000  |
| `global_step_10000/` | 10000 |
| `global_step_15000/` | 15000 |
| `global_step_20000/` | 20000 (final) |

## Training

- Base: LingBot-VLA 2.0 (fp32), 16-dim state/action (14 arm joints + 2 grippers)
- 2×A100-80GB, FSDP2, Muon optimizer, `L1_fm` loss, `bounds_99_woclip` norm, absolute actions
- 20000 steps, lr 1e-4, with depth + DINO-video distillation and future-image prediction

Only the deployable `hf_ckpt` weights are published (optimizer states omitted).
"""
api.upload_file(
    repo_id=REPO,
    path_or_fileobj=readme.encode(),
    path_in_repo="README.md",
    commit_message="Add README",
)
print(f"[{time.strftime('%H:%M:%S')}] ALL DONE — README written", flush=True)
