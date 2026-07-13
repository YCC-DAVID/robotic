#!/usr/bin/env python
# ==============================================================================
# LingBot-VLA 2.0 — Stack-the-cubes 机器人端客户端示例
#
# 在**机器人端**（采集相机 + 下发关节指令的那台机器，通常没有 GPU）运行。
# 通过 websocket 连到推理服务器（deploy/serve_stackcubes.sh 起的那个），
# 发观测、收动作。
#
# 用法:
#   python deploy/client_stackcubes_example.py --host 192.168.1.100 --port 8006
#
# 观测字段（严格按 configs/robot_configs/stackcubes.yaml 的 origin_keys）:
#   observation.images.cam_left_high   : HWC uint8 (H, W, 3)  顶部相机
#   observation.images.cam_left_wrist  : HWC uint8 (H, W, 3)  左腕相机
#   observation.images.cam_right_wrist : HWC uint8 (H, W, 3)  右腕相机
#   observation.state                  : float32 (16,)  14 关节(7左+7右) + 2 夹爪(左,右)
#   task                               : str  语言指令，例如 "stack the cubes"
#
# 返回:
#   result["action"] : chunk_ret=True 时 (chunk, 16)，否则单步 (16,)
#                      前 14 维是双臂关节目标位置，后 2 维是左右夹爪。
# ==============================================================================
import argparse
import time

import numpy as np

try:
    from deploy.websocket_client_policy import WebsocketClientPolicy
except ImportError:  # 直接在 deploy/ 目录下跑
    from websocket_client_policy import WebsocketClientPolicy

ROBO_NAME = "stackcubes"      # 对应 configs/robot_configs/stackcubes.yaml
DEFAULT_TASK = "stack the cubes"


def make_dummy_observation(task: str = DEFAULT_TASK, img_hw=(480, 640)) -> dict:
    """构造一帧假观测，字段名/形状/dtype 与真实部署完全一致，仅用于连通性自测。
    真实部署时把这里的随机图替换成相机帧，state 替换成机器人当前关节+夹爪读数。
    """
    h, w = img_hw
    rand_img = lambda: np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    return {
        "observation.images.cam_left_high": rand_img(),
        "observation.images.cam_left_wrist": rand_img(),
        "observation.images.cam_right_wrist": rand_img(),
        "observation.state": np.zeros(16, dtype=np.float32),
        "task": task,
    }


def main():
    parser = argparse.ArgumentParser(description="Stack-the-cubes real-robot client example")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="推理服务器 IP")
    parser.add_argument("--port", type=int, default=8006, help="推理服务器端口")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="语言指令")
    parser.add_argument("--steps", type=int, default=5, help="发多少帧做自测")
    args = parser.parse_args()

    # 连接（会阻塞等到 server 起来）
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"[client] connected to ws://{args.host}:{args.port}")
    print(f"[client] server metadata: {policy.get_server_metadata()}")

    # 每个新 episode 开始前 reset 一次：加载 robot config + norm stats + feature transform
    policy.reset(ROBO_NAME)
    print(f"[client] reset done (robo_name={ROBO_NAME})")

    # ---- 真实闭环大概长这样 ----
    #   while not done:
    #       obs = {
    #           "observation.images.cam_left_high":  grab_camera("top"),      # HWC uint8
    #           "observation.images.cam_left_wrist": grab_camera("left"),
    #           "observation.images.cam_right_wrist":grab_camera("right"),
    #           "observation.state": robot.get_state_16d().astype(np.float32),
    #           "task": args.task,
    #       }
    #       result = policy.infer(obs)
    #       action_chunk = result["action"]        # (chunk, 16) 若 chunk_ret=True
    #       for a in action_chunk:                 # 逐步下发
    #           robot.command(arm=a[:14], gripper=a[14:16])
    for i in range(args.steps):
        obs = make_dummy_observation(task=args.task)
        t0 = time.time()
        result = policy.infer(obs)
        dt = (time.time() - t0) * 1000
        action = np.asarray(result["action"])
        print(
            f"[client] step {i}: action shape={action.shape}, "
            f"round-trip={dt:.0f}ms, server={result.get('server_timing')}"
        )

    print("[client] done.")


if __name__ == "__main__":
    main()
