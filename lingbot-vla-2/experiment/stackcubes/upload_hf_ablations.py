#!/usr/bin/env python
"""Upload A/B/C ablation checkpoints (deployable hf_ckpt only) to ONE HF repo,
each version under its own subfolder. Idempotent: uploads every currently-ready
step and skips ones already on the hub; does NOT block on steps not yet trained
(re-run this script later to pick up the rest).

  joint100/  <- output/stackcubes_v1      (A: 100ep, joint, vision frozen)
  eef/       <- output/stackcubes_eef      (B: 150ep, EEF,   vision frozen)
  eef100/    <- output/stackcubes_eef100   (C: 100ep, EEF,   vision frozen)
"""
import time
from pathlib import Path
from huggingface_hub import HfApi

REPO = "YccHugAi/lingbot-vla-2-stackcube-ablations"
PRIVATE = False
STEPS = [5000, 10000, 15000, 20000]
ROOT = Path("/scratch/cy65664/workDir/lingbot-vla-v2/output")
VERSIONS = {
    "joint100": ROOT / "stackcubes_v1"     / "checkpoints",
    "eef":      ROOT / "stackcubes_eef"     / "checkpoints",
    "eef100":   ROOT / "stackcubes_eef100"  / "checkpoints",
}

api = HfApi()


def ckpt_dir(base: Path, step: int) -> Path:
    return base / f"global_step_{step}"


def ready(base: Path, step: int) -> bool:
    """hf_ckpt fully written: index + 6 shards, no unfinished tmp sibling, size-stable 30s."""
    cdir = ckpt_dir(base, step)
    d = cdir / "hf_ckpt"
    if not d.is_dir():
        return False
    if list(cdir.parent.glob(".hf_ckpt.tmp.*")):   # async writer still finalizing
        return False
    idx = d / "model.safetensors.index.json"
    shards = list(d.glob("model-*-of-*.safetensors"))
    if not idx.exists() or len(shards) < 6:
        return False
    sizes = {s: s.stat().st_size for s in shards}
    time.sleep(30)
    return all(s.exists() and s.stat().st_size == sz for s, sz in sizes.items())


def on_hub(files, subdir: str, step: int) -> bool:
    return any(f.startswith(f"{subdir}/global_step_{step}/model.safetensors.index.json") for f in files)


def main():
    try:
        api.repo_info(REPO)
    except Exception:
        api.create_repo(REPO, private=PRIVATE, repo_type="model", exist_ok=True)
        print(f"created repo {REPO} (private={PRIVATE})", flush=True)

    try:
        hub_files = api.list_repo_files(REPO)
    except Exception:
        hub_files = []

    uploaded, pending, skipped = [], [], []
    for subdir, base in VERSIONS.items():
        for step in STEPS:
            tag = f"{subdir}/global_step_{step}"
            if on_hub(hub_files, subdir, step):
                skipped.append(tag); continue
            if not ready(base, step):
                pending.append(tag); continue
            src = ckpt_dir(base, step) / "hf_ckpt"
            print(f"[{time.strftime('%H:%M:%S')}] >>> upload {tag} ({src})", flush=True)
            api.upload_folder(
                repo_id=REPO, folder_path=str(src), path_in_repo=tag,
                commit_message=f"Add {tag} (deployable hf_ckpt)",
            )
            uploaded.append(tag)
            print(f"[{time.strftime('%H:%M:%S')}] <<< done {tag}", flush=True)

    readme = """---
license: apache-2.0
library_name: transformers
tags: [robotics, vla, lingbot-vla, lerobot, manipulation]
---

# LingBot-VLA 2.0 — Stack-the-cubes ablations

Finetunes of **LingBot-VLA 2.0** (6B, MoE action expert) on Unitree G1 Dex1 dual-arm
cube stacking. All runs freeze the vision encoder; distillation teachers stay frozen,
current+future depth/DINO query heads train. Muon, lr 1e-4, L1_fm, bounds_99_woclip,
absolute actions, 20000 steps, 2xA100.

| Subfolder | Data | Action space | Note |
|---|---|---|---|
| `joint100/` | 100ep | joint (14 arm + 2 grip) | vision frozen |
| `eef/`      | 150ep (v1+v2) | EEF pose (14 + 2 grip) | vision frozen |
| `eef100/`   | 100ep | EEF pose (14 + 2 grip) | vision frozen |

Ablation axes: `joint100` vs `eef100` = joint vs EEF (same data); `eef` vs `eef100`
= mixed 150ep vs clean 100ep. Each `*/global_step_*/` holds deployable HF weights
(`model-0000x-of-00006.safetensors` + tokenizer/config). EEF models output end-effector
poses — the client must IK them back to joints (see g1-client `main_eef.py`).
"""
    api.upload_file(repo_id=REPO, path_or_fileobj=readme.encode(),
                    path_in_repo="README.md", commit_message="Update README")

    print("\n===== SUMMARY =====", flush=True)
    print("uploaded:", uploaded or "(none)", flush=True)
    print("already on hub:", skipped or "(none)", flush=True)
    print("pending (not ready / not trained yet):", pending or "(none)", flush=True)


if __name__ == "__main__":
    main()
