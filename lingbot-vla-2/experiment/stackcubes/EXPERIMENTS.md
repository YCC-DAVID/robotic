# Stack-the-cubes 实验记录 (LingBot-VLA 2.0 finetune)

> 纯实验条件记录。每挂一版训练在此追加一行/一段。不放权重、不放大 log,只记"跑了什么、怎么跑的、结果如何"。
> 模型: LingBot-VLA 2.0 6B (MoE action expert, flow-matching L1_fm, depth+DINO-video 蒸馏, future-image 预测)。
> 机器人: Unitree G1 Dex1,16 维 state/action = 14 臂关节(7L+7R) + 2 夹爪,3 相机 (camera_top / wrist_left / wrist_right),无下半身。

## 数据集

| 名称 | episodes | frames | 空间 | prompt |
|---|---|---|---|---|
| `Stack-the-cubes`        | 100 | 95966  | joint | "Stack the blocks by color: put the red block in the center, then stack the blue block on the red block, then stack the yellow block on the blue block." |
| `Stack-the-cubes-v2`     | 50  | 42344  | joint | "pick up cube." |
| `Stack-the-cubes-eef`    | 100 | 95966  | EEF   | (同 Stack-the-cubes 完整句) |
| `Stack-the-cubes-v2-eef` | 50  | 42344  | EEF   | "pick up cube." ⚠️ 与主任务描述不一致 |

- EEF 数据由 `scripts/joint_to_eef.py`(= 对方 openpi-fintune 原版脚本,仅改 URDF 路径常量)做 pinocchio FK 得到。
  URDF `assets/g1/g1_body29_hand14.urdf` 与 xr_teleoperate 那份 md5 逐字节一致 (`b2acaa06…`),mesh 全一致。
- EEF 16 维 = [L pose7(x,y,z,qx,qy,qz,qw) | R pose7 | L grip | R grip],pelvis 帧,四元数 xyzw,**绝对位姿** (subtract_state=False,对齐 openpi `main_eef.py`)。

## 固定超参 (除非注明,各版一致)

- optimizer muon, lr 1e-4 constant, loss L1_fm
- micro_batch 1, grad_accum 1, GBS = GPU 数 (2×A100 → GBS=2)
- max_steps 20000, save_steps 5000
- action_dim / max_action_dim / max_state_dim = 55 (16 维有效,其余 padding 被 action_joint_mask 屏蔽,不算 loss)
- tokenizer_max_length 72, prompt_type global, Qwen3-VL chat 模板
- MoE: 36 层 token-MoE, 32 experts, top-4；align: depth(current+future) + future-video(DINO) 蒸馏
- **冻结范围**: vision encoder (Qwen3-VL 视觉塔) + 蒸馏 teacher(MoGe/MoRGBD/DINO,永远 .eval());
  current/future 的 depth & video align query + head **均训练** (requires_grad=True)。
- 单版 20000 步实测 ≈ 14–15h (2 卡 A100, ~2.5 s/it)。

## 实验清单

| 版本 | job (norm→train) | 数据 | 空间 | vision enc | 监督目标 (norm key) | output_dir | 状态 |
|---|---|---|---|---|---|---|---|
| **Baseline** | — (周末) | v1+v2 (150ep) | joint | **全 finetune** | arm.position | output/stackcubes | ✅ 完成 7/13 12:35 |
| **A** | 46954620 → 46954621 | v1 (100ep) | joint | 冻结 | arm.position | output/stackcubes_v1 | ▶ RUNNING (7/14 收尾) |
| **B** | 46954961 → 46954962 | v1+v2 (150ep) | **EEF** | 冻结 | end.position | output/stackcubes_eef | ▶ RUNNING (7/14 收尾) |
| **C** | 46961233 → 46961234 | v1 (100ep) | **EEF** | 冻结 | end.position | output/stackcubes_eef100 | ✅ 完成 7/15 01:13 (14h04m) |

### 对比设计
- **A vs Baseline**: 冻结 vs 全 finetune + 100ep vs 150ep(混合是否反而更差)。
- **A vs C**: 同 100ep、同冻结,唯一变量 = joint vs EEF 动作空间(验证"只学 EEF 真机更好")。
- **B vs C**: 同 EEF、同冻结,唯一变量 = 150ep 混合 vs 100ep 纯净(v2 的 "pick up cube" prompt 不一致是否拖累混合)。

### 备注 / 坑
- ⚠️ **B 的 prompt 不一致**: 混了 100ep 完整句 + 50ep "pick up cube",language grounding 被稀释。若 v2 实为叠方块,应统一 prompt 后重训 B。C 无此问题。
- norm_stats 路径由 **robot_config 的 `norm_stats:` 字段** 决定(不是 data.norm_stats_file),每版需独立 robot_config + 独立 norm json。
- max_steps=20000 是真实终点;日志 INFO 行 `/47983`、`/69155` 是"一个 epoch 的帧数",被 max_steps 截断,跑不到。
- ⚠️ 数据集 meta 里 `robot_type: Unitree_G1_Dex1_Sim` 是**误标** —— 数据为真机采集(转换脚本写死的标签,训练不读该字段)。因此不存在可用的闭环仿真;LIBERO/RoboTwin 均不匹配 G1 双臂,仿真评测无意义。

## 真机抽搐归因 (7/16 离线测试, Version B @20000, episode 0)

- **IK 无辜**: GT EEF → warm-start DLS IK → 关节,重建误差 0.37-0.78°,残差 0.008mm,0 抽搐帧,加速度 1.00x GT (`ik_roundtrip_test.py`, 4 episodes)。
- **模型输出本身抖 = 主犯**: 喂录制帧给部署 policy (`model_twitch_test.py`, job 47067456):
  - A 块内 EEF 加速度 RMS **17.7x GT** (2530 vs 143 mm/s²),~mm 级 30Hz 毛刺
  - B 跟踪正常: pos 15.7mm / quat 3.9° (方向对,只是叠噪声)
  - C 接缝跳变 mean 23.4mm / max 59mm (时间对齐后仍存在 = 重规划不一致,掐头治不了,只能 blend/滤波掩盖)
  - D IK 后关节加速度 **36.5x GT**,>6°/帧抽搐 **21 帧/22.6s** (GT 0 帧) —— 真机症状离线复现
- **对策 + A/B 实测** (job 47067456 基线 vs 47069019 savgol w=7,同 episode 同 stride,基线独立复跑数字逐位一致=测量稳定):
  | 指标 | 原始 | savgol | 说明 |
  |---|---|---|---|
  | 块内 EEF acc (xGT) | 17.7x | **3.7x** | 压 4.8 倍 |
  | 跟踪 pos/quat | 15.7mm/3.9° | **15.7mm/3.9°** | 不变——滤波不掰轨迹 |
  | IK 后关节 acc (xGT) | 36.5x | 24.4x | 剩余来自接缝 |
  | 抽搐帧 (>6°/帧, 无blend硬拼) | 21 | 17 | savgol 只删掉 4 帧 → 剩余 17 帧≈27 个接缝处,归 client blend 管 |
  | 接缝跳变 | 23.4mm | 23.4mm | 滤波管不了,预期内 |
  结论: **块内毛刺已由 server savgol 解决;残余抽搐 = 接缝**,真机上开 `--blend-steps 10` + `--exec-steps 25` 即被渐变掩盖(测试的 D 项故意不加 blend 以显示最坏情况)。client 还需 `--control-hz 30`(默认 15 是错的)+ `--velocity-limit 2-3` 兜底。图: `model_twitch_ep0_{raw,savgol}.png`。
- **serving 栈修复(同日)**: openpi 客户端契约兼容(自动 reset/`prompt`→`task`/返回 `actions` 别名/JPEG bytes 解码)、GPU-first 权重加载(旧 CPU 路径在慢节点 >20min 且全核 AVX 可致工作站热节流降频)、`ROBO_NAME` 与 MODEL_PATH 错配 WARN。

## 部署 prompt

叠方块任务统一给完整句:
> Stack the blocks by color: put the red block in the center, then stack the blue block on the red block, then stack the yellow block on the blue block.

- joint 版 (A/Baseline): 模型直接吐关节,客户端直接下发。
- EEF 版 (B/C): 模型吐 EEF 位姿,客户端需 IK 回关节 (`g1-client/openpi/main_eef.py` + `eef_kinematics.py`,warm-start damped-LS)。

## 真机结果 (待填)

| 版本 | 成功率 | 备注 |
|---|---|---|
| A |  |  |
| B |  |  |
| C |  |  |
