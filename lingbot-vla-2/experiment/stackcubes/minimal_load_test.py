#!/usr/bin/env python
"""Load-path diagnosis v2.

v1 finding: load_state_dict (CPU, fp32 model <- bf16 shards, single thread)
ran >24 min without finishing — the CPU detour is pathologically slow on
these nodes. v2 tests the GPU-first path:
  construct (cpu fp32) -> .to(bf16) -> .cuda() -> read shards straight to
  GPU -> copy GPU->GPU.
Each stage timed; faulthandler dumps every 3 min in case anything stalls.
"""
import sys, time, faulthandler
faulthandler.dump_traceback_later(180, repeat=True, file=sys.stderr)

sys.path.insert(0, "/scratch/cy65664/workDir/lingbot-vla-v2")
T0 = time.time()
def mark(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

mark("importing torch ...")
import torch
mark(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}, "
     f"threads={torch.get_num_threads()}")

mark("importing deploy module ...")
from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server, apply_lingbot_qwen3_vl_patch
mark("deploy module imported; applying patch ...")
apply_lingbot_qwen3_vl_patch()

MODEL = "/scratch/cy65664/workDir/lingbot-vla-v2/output/stackcubes_eef/checkpoints/global_step_20000/hf_ckpt"

srv = LingbotVLAv2Server.__new__(LingbotVLAv2Server)
srv.adaptive_ensemble_alpha = 0.1
srv.action_ensemble_horizon = 8
srv.use_length = 50
srv.chunk_ret = True
srv.robot_norm_path = None
srv.task_description = None
srv.use_compile = False
srv.default_robo_name = "stackcubes_eef"

# GPU-first weight loading: skip the CPU fp32 detour entirely.
def gpu_load_weights(self, path, strict=True):
    import os
    from glob import glob
    from safetensors.torch import load_file
    mark("  moving empty model to bf16 ...")
    self.vla = self.vla.to(torch.bfloat16)
    mark("  moving empty model to cuda ...")
    self.vla = self.vla.cuda()
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    merged = {}
    for fp in files:
        t = time.time()
        merged.update(load_file(fp, device="cuda"))
        mark(f"  shard -> GPU: {os.path.basename(fp)} ({time.time()-t:.1f}s)")
    mark(f"  load_state_dict on GPU ({len(merged)} tensors, strict={strict}) ...")
    t = time.time()
    self.vla.load_state_dict(merged, strict=strict)
    mark(f"  load_state_dict DONE ({time.time()-t:.1f}s)")
LingbotVLAv2Server.load_model_weights = gpu_load_weights

mark("load_vla (GPU-first) ...")
vla = srv.load_vla(MODEL)
vla = vla.eval()
mark("load_vla DONE")

# smoke: one dummy forward through sample_actions via the real infer path
import numpy as np
srv.vla = vla
srv.use_bf16, srv.use_fp32 = True, False
srv.global_step, srv.last_action_chunk, srv.last_normalized_action_chunk = 0, None, None
srv.action_key = "action"
srv.sample_actions_fn = vla.model.sample_actions
mark("reset (feature transform) ...")
import os
os.chdir("/scratch/cy65664/workDir/lingbot-vla-v2")
srv.reset(robo_name="stackcubes_eef")
mark("reset DONE; dummy infer ...")
obs = {
    "observation.images.cam_left_high": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
    "observation.images.cam_left_wrist": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
    "observation.images.cam_right_wrist": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
    "observation.state": np.zeros(16, dtype=np.float32),
    "prompt": "stack the cubes",
}
t = time.time()
r = srv.infer(obs)
a = np.asarray(r["actions"])
mark(f"dummy infer DONE ({time.time()-t:.1f}s), actions shape={a.shape}")
t = time.time()
r = srv.infer(obs)
mark(f"second infer DONE ({time.time()-t:.1f}s)")
mark("ALL STAGES PASSED, no hang")
faulthandler.cancel_dump_traceback_later()
