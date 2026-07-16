#!/usr/bin/env python
"""Open-loop MODEL twitch test — feeds recorded dataset frames to the actual
deployment policy and inspects the EEF action chunks for jitter, WITHOUT a robot.

Reuses deploy/lingbot_vla_v2_policy.LingbotVLAv2Server exactly as the websocket
server does (same normalization / model / unnormalization), so the EEF actions
here are byte-identical to what the robot would receive.

Four diagnostics:
  A. within-chunk jitter  — is a single predicted 50-step chunk smooth?
  B. vs ground-truth      — does the chunk track the demo (pos mm / quat deg)?
  C. cross-chunk seam     — replan at t and t+stride: does the new chunk agree
                            with the old one where they overlap? (source of 动作回抽)
  D. post-IK joint jitter — IK the model EEF the way main_eef does, count twitch
                            frames in JOINT space (the real on-robot twitch metric)
"""
import os, sys, argparse, time
import faulthandler
# The load path has hung twice (futex_wait, ~35min, after safetensors read) on
# two different nodes — dump every thread's stack every 5 min so the hang
# point is visible in the log instead of a silent stall.
faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)
import numpy as np
import pandas as pd

REPO = "/scratch/cy65664/workDir/lingbot-vla-v2"
sys.path.insert(0, REPO)
sys.path.insert(0, "/scratch/cy65664/workDir/g1-client/openpi")

DS   = f"{REPO}/datasets/Stack-the-cubes-eef"
PARQUET = f"{DS}/data/chunk-000/file-000.parquet"
EP_META = f"{DS}/meta/episodes/chunk-000/file-000.parquet"
FPS = 30.0
CAMS = ["observation.images.cam_left_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist"]
LEFT, RIGHT = slice(0, 7), slice(7, 14)


def decode_episode_frames(ep):
    """Return {cam: (T,H,W,3) uint8} for one episode, plus GT state/action (T,16)."""
    import av
    meta = pd.read_parquet(EP_META)
    m = meta[meta.episode_index == ep].iloc[0]
    df = pd.read_parquet(PARQUET, columns=["observation.state", "action", "episode_index", "frame_index"])
    d = df[df.episode_index == ep].sort_values("frame_index")
    state = np.stack(d["observation.state"].to_numpy()).astype(np.float32)
    action = np.stack(d["action"].to_numpy()).astype(np.float32)
    T = len(d)

    frames = {}
    for cam in CAMS:
        ci = int(m[f"videos/{cam}/chunk_index"]); fi = int(m[f"videos/{cam}/file_index"])
        t0 = float(m[f"videos/{cam}/from_timestamp"])
        path = f"{DS}/videos/{cam}/chunk-{ci:03d}/file-{fi:03d}.mp4"
        container = av.open(path)
        stream = container.streams.video[0]
        # seek near episode start, then collect T frames from t0
        container.seek(int(t0 / stream.time_base), stream=stream, any_frame=False, backward=True)
        buf = []
        for f in container.decode(stream):
            if f.time is None:
                continue
            if f.time < t0 - 1e-3:
                continue
            buf.append(f.to_ndarray(format="rgb24"))
            if len(buf) >= T:
                break
        container.close()
        if len(buf) < T:                      # pad by repeating last frame if short
            buf += [buf[-1]] * (T - len(buf))
        frames[cam] = np.stack(buf[:T])
        print(f"  decoded {cam}: {frames[cam].shape}", flush=True)
    return frames, state, action, T


def quat_norm(a):
    a = a.copy()
    for sl in (LEFT, RIGHT):
        q = a[..., sl][..., 3:7]
        a[..., sl.start + 3:sl.start + 7] = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)
    return a


def quat_geodesic_deg(q1, q2):
    d = np.abs((q1 * q2).sum(-1)).clip(0, 1)
    return np.degrees(2 * np.arccos(d))


def pos_jitter_mm(chunk, dt):
    """Per-step L/R EEF position speed & accel of a (K,16) chunk, in mm."""
    pos = np.concatenate([chunk[:, 0:3], chunk[:, 7:10]], axis=1) * 1000.0  # (K,6) mm
    vel = np.diff(pos, axis=0) / dt
    acc = np.diff(pos, axis=0, n=2) / dt**2
    return dict(vel_rms=np.sqrt((vel**2).mean()), acc_rms=np.sqrt((acc**2).mean()),
                step_max=np.abs(np.diff(pos, axis=0)).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=f"{REPO}/output/stackcubes_eef/checkpoints/global_step_20000/hf_ckpt")
    ap.add_argument("--robo_name", default="stackcubes_eef")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--stride", type=int, default=25, help="replan interval (frames) for seam test")
    ap.add_argument("--use-length", type=int, default=50)
    ap.add_argument("--prompt", default="Stack the blocks by color: put the red block in the center, then stack the blue block on the red block, then stack the yellow block on the blue block.")
    ap.add_argument("--outdir", default=f"{REPO}/experiment/stackcubes")
    ap.add_argument("--twitch-thresh", type=float, default=0.10, help="joint jump (rad) counted as twitch")
    ap.add_argument("--smooth", default="none", choices=["none", "savgol", "ema"],
                    help="server-side chunk smoothing to A/B test against raw output")
    ap.add_argument("--smooth-window", type=int, default=7)
    args = ap.parse_args()
    dt = 1.0 / FPS

    # Load the policy BEFORE touching PyAV/FFmpeg: importing av and decoding
    # first left load_state_dict deadlocked on a futex (duelling OpenMP
    # runtimes). Torch/model first, video decode second is the safe order.
    print(f"[{time.strftime('%H:%M:%S')}] loading policy ...", flush=True)
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    model = LingbotVLAv2Server(args.model_path, use_length=args.use_length,
                               chunk_ret=True, use_bf16=True, use_fp32=False, use_compile=False,
                               smooth=args.smooth, smooth_window=args.smooth_window)
    model.infer({"reset": True, "robo_name": args.robo_name, "path_to_pi_model": args.model_path})
    print(f"[{time.strftime('%H:%M:%S')}] policy ready", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] decoding episode {args.episode} ...", flush=True)
    frames, gt_state, gt_action, T = decode_episode_frames(args.episode)
    print(f"  T={T} frames ({T/FPS:.1f}s)", flush=True)

    query_t = list(range(0, T - 1, args.stride))
    chunks = {}
    for t in query_t:
        obs = {CAMS[0]: frames[CAMS[0]][t], CAMS[1]: frames[CAMS[1]][t], CAMS[2]: frames[CAMS[2]][t],
               "observation.state": gt_state[t].copy(), "prompt": args.prompt}
        r = model.infer(obs)
        chunks[t] = quat_norm(np.asarray(r["action"], dtype=np.float64))   # (chunk,16), quats unit-normed
        print(f"  [{time.strftime('%H:%M:%S')}] chunk@t={t}: {chunks[t].shape}", flush=True)

    K = chunks[query_t[0]].shape[0]

    # ---------- A. within-chunk jitter vs GT ----------
    print("\n=== A. within-chunk EEF position jitter (mm) ===")
    print(f"  {'t':>5} {'model vel_rms':>14} {'GT vel_rms':>12} {'model acc_rms':>14} {'GT acc_rms':>12} {'model step_max':>15}")
    mv=[]; ma=[]; gv=[]; ga=[]
    for t in query_t:
        h = min(K, T - t)
        mj = pos_jitter_mm(chunks[t][:h], dt)
        gj = pos_jitter_mm(quat_norm(gt_action[t:t+h].astype(np.float64)), dt)
        mv.append(mj["vel_rms"]); ma.append(mj["acc_rms"]); gv.append(gj["vel_rms"]); ga.append(gj["acc_rms"])
        print(f"  {t:5d} {mj['vel_rms']:14.2f} {gj['vel_rms']:12.2f} {mj['acc_rms']:14.1f} {gj['acc_rms']:12.1f} {mj['step_max']:15.2f}")
    print(f"  MEAN  model acc_rms={np.mean(ma):.1f}  GT acc_rms={np.mean(ga):.1f}  -> ratio {np.mean(ma)/max(np.mean(ga),1e-9):.2f}x")

    # ---------- B. vs GT tracking ----------
    print("\n=== B. chunk vs GT (does model reproduce the demo) ===")
    pos_err=[]; quat_err=[]
    for t in query_t:
        h = min(K, T - t)
        c = chunks[t][:h]; g = quat_norm(gt_action[t:t+h].astype(np.float64))
        pe = np.linalg.norm(np.concatenate([c[:,0:3]-g[:,0:3], c[:,7:10]-g[:,7:10]],axis=1).reshape(-1,3),axis=1)*1000
        qe = np.concatenate([quat_geodesic_deg(c[:,3:7],g[:,3:7]), quat_geodesic_deg(c[:,10:14],g[:,10:14])])
        pos_err.append(pe.mean()); quat_err.append(np.nanmean(qe))
    print(f"  pos err  mean={np.mean(pos_err):.1f}mm   quat err mean={np.nanmean(quat_err):.1f}deg")

    # ---------- C. cross-chunk seam (动作回抽 source) ----------
    print("\n=== C. cross-chunk seam discontinuity at replan ===")
    print("  (position the robot jumps to when a NEW chunk replaces the old, at the seam)")
    seams=[]
    for a, b in zip(query_t[:-1], query_t[1:]):
        off = b - a
        if off >= K: continue
        old = chunks[a][off]      # where old chunk says we are at replan time b
        new = chunks[b][0]        # where new chunk starts
        jump = np.linalg.norm(np.concatenate([new[0:3]-old[0:3], new[7:10]-old[7:10]]).reshape(-1,3),axis=1)*1000
        seams.append(jump.max())
    if seams:
        print(f"  seam jump (max of L/R): mean={np.mean(seams):.1f}mm  max={np.max(seams):.1f}mm  p90={np.percentile(seams,90):.1f}mm")
        print(f"  (main_eef mitigates this via --chunk-align + --blend-steps; large values here = why blending is needed)")

    # ---------- D. post-IK joint jitter (real twitch metric) ----------
    print("\n=== D. post-IK JOINT jitter — IK the model EEF like main_eef ===")
    from eef_kinematics import G1DualArmKinematics
    kin = G1DualArmKinematics()
    # stitch model chunks into one open-loop joint trajectory (naive concat at stride, no blend)
    stitched = []
    for a, b in zip(query_t[:-1], query_t[1:] + [T]):
        n = min(b - a, K)
        stitched.append(chunks[a][:n])
    stitched = np.concatenate(stitched, axis=0)[:T]
    ik_q = gt_state[0, :14] if gt_state.shape[1] >= 14 else np.zeros(14)
    # eef model state is EEF pose, so warm-start IK from GT joints of the JOINT dataset frame 0
    jdf = pd.read_parquet(f"{REPO}/datasets/Stack-the-cubes/data/chunk-000/file-000.parquet",
                          columns=["action","episode_index","frame_index"])
    jd = jdf[jdf.episode_index==args.episode].sort_values("frame_index")
    gt_joints = np.stack(jd["action"].to_numpy()).astype(np.float64)[:T, :14]
    ik_q = gt_joints[0].copy()
    rec = np.zeros((len(stitched),14)); perr=np.zeros(len(stitched))
    for i,a in enumerate(stitched):
        ik_q,e = kin.solve_ik(a[LEFT], a[RIGHT], ik_q); rec[i]=ik_q; perr[i]=e
    def jj(x):
        v=np.diff(x,axis=0)/dt; ac=np.diff(x,axis=0,n=2)/dt**2
        return np.sqrt((v**2).mean()), np.sqrt((ac**2).mean()), np.abs(np.diff(x,axis=0)).max()
    gv2,ga2,gs2 = jj(gt_joints); mv2,ma2,ms2 = jj(rec)
    jump = np.abs(np.diff(rec,axis=0)); gjump=np.abs(np.diff(gt_joints,axis=0))
    m_tw=int((jump.max(1)>args.twitch_thresh).sum()); g_tw=int((gjump.max(1)>args.twitch_thresh).sum())
    print(f"  joint acc_rms: model-IK={ma2:.2f}  GT={ga2:.2f}  ratio={ma2/max(ga2,1e-9):.2f}x")
    print(f"  joint step_max: model-IK={np.degrees(ms2):.1f}deg  GT={np.degrees(gs2):.1f}deg")
    print(f"  twitch frames (>{np.degrees(args.twitch_thresh):.0f}deg/frame): model-IK={m_tw}/{len(rec)-1}  GT={g_tw}/{len(gt_joints)-1}")
    print(f"  IK residual on model EEF: mean={perr.mean()*1000:.2f}mm  max={perr.max()*1000:.2f}mm  (large=model EEF unreachable)")

    # ---------- verdict ----------
    print("\n=== VERDICT ===")
    twitchy = (m_tw > g_tw + 5) or (ma2/max(ga2,1e-9) > 3.0) or (np.mean(ma)/max(np.mean(ga),1e-9) > 3.0)
    if twitchy:
        print(f"  MODEL OUTPUT IS JITTERY: EEF acc {np.mean(ma)/max(np.mean(ga),1e-9):.1f}x GT, "
              f"post-IK joint acc {ma2/max(ga2,1e-9):.1f}x GT, {m_tw} twitch frames (GT {g_tw}).")
        print("  -> The twitching originates in the MODEL's action output, not the IK.")
    else:
        print(f"  MODEL OUTPUT IS SMOOTH within chunks (EEF acc {np.mean(ma)/max(np.mean(ga),1e-9):.1f}x GT, "
              f"{m_tw} twitch frames vs GT {g_tw}).")
        if seams and np.mean(seams) > 8:
            print(f"  BUT cross-chunk seams jump {np.mean(seams):.0f}mm on replan -> twitching is the STITCHING;")
            print("  ensure --chunk-align + --blend-steps are on (and consider --exec-steps to trim drift).")
        else:
            print("  and seams are small. If the robot still twitches, suspect control-rate / comms / state feedback.")

    # ---------- plots ----------
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,1,figsize=(13,9))
        tt=np.arange(T)/FPS
        # left EEF x/y/z: GT vs stitched-model
        for k,lbl in [(0,"Lx"),(1,"Ly"),(2,"Lz")]:
            axes[0].plot(tt, gt_action[:T,k]*1000, lw=1.4)
            axes[0].plot(np.arange(len(stitched))/FPS, stitched[:,k]*1000, lw=0.9, ls="--")
        axes[0].set_title("Left EEF pos (mm): solid=GT, dashed=model (stitched)"); axes[0].grid(alpha=.3); axes[0].set_ylabel("mm")
        # per-chunk overlay to see within-chunk smoothness (left x)
        for t in query_t:
            h=min(K,T-t); axes[1].plot(np.arange(t,t+h)/FPS, chunks[t][:h,0]*1000, lw=0.8)
        axes[1].plot(tt, gt_action[:T,0]*1000, "k", lw=1.6, alpha=.5, label="GT")
        axes[1].set_title("Left EEF x: each model chunk (colored) vs GT (black) — seams visible at overlaps")
        axes[1].grid(alpha=.3); axes[1].legend(); axes[1].set_ylabel("mm")
        # post-IK joints
        for j in [0,3,6]:
            axes[2].plot(np.arange(len(gt_joints))/FPS, np.degrees(gt_joints[:,j]), lw=1.4)
            axes[2].plot(np.arange(len(rec))/FPS, np.degrees(rec[:,j]), lw=0.9, ls="--")
        axes[2].set_title(f"Post-IK joints j0/j3/j6 (deg): solid=GT, dashed=model-IK  ({m_tw} twitch frames)")
        axes[2].grid(alpha=.3); axes[2].set_xlabel("time (s)"); axes[2].set_ylabel("deg")
        fig.suptitle(f"Model twitch test — {args.robo_name} ep{args.episode}, stride={args.stride}")
        fig.tight_layout()
        out=f"{args.outdir}/model_twitch_ep{args.episode}.png"; fig.savefig(out,dpi=110)
        print(f"\nplot -> {out}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
