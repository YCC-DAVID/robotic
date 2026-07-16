#!/usr/bin/env python
"""IK round-trip twitch test — NO model, NO GPU.

Isolates whether the on-robot twitching comes from the IK (main_eef's
damped-least-squares solve_ik) rather than the model or chunk-stitching.

Method: take PERFECTLY SMOOTH ground-truth EEF poses (human teleop, FK'd by
joint_to_eef.py) from the eef dataset, run them through the *exact same*
per-step warm-started IK that main_eef uses, and compare the recovered joint
trajectory against the ground-truth joints (from the original joint dataset).

If clean GT EEF -> IK produces jittery joints, the IK is the culprit.
If recovered joints track GT joints smoothly, twitching is elsewhere
(model output or chunk stitching).
"""
import sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, "/scratch/cy65664/workDir/g1-client/openpi")
from eef_kinematics import G1DualArmKinematics
LEFT_EEF_CHANNELS  = slice(0, 7)   # [L xyz+quat] — same as main_eef.py
RIGHT_EEF_CHANNELS = slice(7, 14)  # [R xyz+quat]

EEF_DS   = "/scratch/cy65664/workDir/lingbot-vla-v2/datasets/Stack-the-cubes-eef/data/chunk-000/file-000.parquet"
JOINT_DS = "/scratch/cy65664/workDir/lingbot-vla-v2/datasets/Stack-the-cubes/data/chunk-000/file-000.parquet"
FPS = 30.0


def load_ep(path, ep, col):
    df = pd.read_parquet(path, columns=[col, "episode_index", "frame_index"])
    d = df[df.episode_index == ep].sort_values("frame_index")
    return np.stack(d[col].to_numpy())  # (T, 16)


def jerk_stats(traj, dt):
    """Per-step velocity + acceleration magnitude of a (T,D) trajectory."""
    vel = np.diff(traj, axis=0) / dt                    # (T-1, D)
    acc = np.diff(traj, axis=0, n=2) / dt**2            # (T-2, D)
    return dict(
        vel_rms=np.sqrt((vel**2).mean()),
        vel_max=np.abs(vel).max(),
        acc_rms=np.sqrt((acc**2).mean()),
        acc_max=np.abs(acc).max(),
        step_max=np.abs(np.diff(traj, axis=0)).max(),   # biggest single-frame jump
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole episode")
    ap.add_argument("--twitch-thresh", type=float, default=0.10,
                    help="per-frame joint jump (rad) counted as a twitch")
    ap.add_argument("--plot", default="/scratch/cy65664/workDir/lingbot-vla-v2/experiment/stackcubes/ik_roundtrip.png")
    args = ap.parse_args()

    eef_gt   = load_ep(EEF_DS,   args.episode, "action")            # smooth GT EEF (what an ideal model would output)
    joint_gt = load_ep(JOINT_DS, args.episode, "action")            # smooth GT joints (14 arm + 2 grip)
    T = min(len(eef_gt), len(joint_gt))
    if args.max_frames:
        T = min(T, args.max_frames)
    eef_gt, joint_gt = eef_gt[:T], joint_gt[:T]
    print(f"episode {args.episode}: {T} frames @ {FPS}fps ({T/FPS:.1f}s)", flush=True)

    kin = G1DualArmKinematics()

    # warm-start from GT joints at frame 0, then chain from previous IK output — exactly like main_eef
    ik_q = joint_gt[0, :14].astype(np.float64).copy()
    rec = np.zeros((T, 14))
    pos_err = np.zeros((T, 2))
    for t in range(T):
        ik_q, err = kin.solve_ik(eef_gt[t, LEFT_EEF_CHANNELS], eef_gt[t, RIGHT_EEF_CHANNELS], ik_q)
        rec[t] = ik_q
        pos_err[t] = err

    dt = 1.0 / FPS
    gt_arm = joint_gt[:, :14]

    # --- reconstruction accuracy (FK/IK self-consistency) ---
    recon = np.abs(rec - gt_arm)
    print("\n=== IK vs GT joints (reconstruction) ===")
    print(f"  joint abs err : mean={np.degrees(recon.mean()):.3f}deg  "
          f"max={np.degrees(recon.max()):.3f}deg  p99={np.degrees(np.percentile(recon,99)):.3f}deg")
    print(f"  IK pos resid  : mean={pos_err.mean()*1000:.3f}mm  max={pos_err.max()*1000:.3f}mm")

    # --- jitter comparison ---
    g = jerk_stats(gt_arm, dt)
    r = jerk_stats(rec, dt)
    print("\n=== joint-space jitter: GT vs IK-recovered ===")
    print(f"  {'metric':12s} {'GT':>12s} {'IK':>12s}  {'ratio':>7s}")
    for k in ["vel_rms", "vel_max", "acc_rms", "acc_max", "step_max"]:
        u = "rad/s" if "vel" in k else ("rad/s^2" if "acc" in k else "rad")
        print(f"  {k:12s} {g[k]:12.4f} {r[k]:12.4f}  {r[k]/max(g[k],1e-9):6.2f}x  ({u})")

    # --- twitch count: per-frame joint jumps above threshold ---
    gt_jump = np.abs(np.diff(gt_arm, axis=0))
    ik_jump = np.abs(np.diff(rec, axis=0))
    gt_tw = int((gt_jump.max(axis=1) > args.twitch_thresh).sum())
    ik_tw = int((ik_jump.max(axis=1) > args.twitch_thresh).sum())
    print(f"\n=== twitch frames (any joint jumps > {args.twitch_thresh} rad = "
          f"{np.degrees(args.twitch_thresh):.1f}deg in 1/{int(FPS)}s) ===")
    print(f"  GT: {gt_tw}/{T-1} frames    IK: {ik_tw}/{T-1} frames")
    if ik_jump.size:
        wj = ik_jump.max(axis=1)
        worst = np.argsort(wj)[-5:][::-1]
        print("  worst IK jumps (frame -> joint / deg):")
        for f in worst:
            j = ik_jump[f].argmax()
            print(f"    frame {f:4d}: joint {j:2d}  {np.degrees(wj[f]):6.2f}deg  (GT here {np.degrees(gt_jump[f].max()):5.2f}deg)")

    # --- verdict ---
    ratio = r["acc_rms"] / max(g["acc_rms"], 1e-9)
    print("\n=== VERDICT ===")
    if ik_tw > gt_tw + 2 or ratio > 3.0:
        print(f"  IK INTRODUCES JITTER (acc {ratio:.1f}x GT, {ik_tw-gt_tw} extra twitch frames).")
        print("  -> On-robot twitching is at least partly the IK's fault.")
    elif recon.mean() > np.radians(2.0):
        print(f"  IK doesn't track GT well ({np.degrees(recon.mean()):.2f}deg mean) — solver/config issue.")
    else:
        print(f"  IK is SMOOTH & accurate (acc {ratio:.1f}x GT, {ik_tw} twitch frames, "
              f"{np.degrees(recon.mean()):.2f}deg err).")
        print("  -> Twitching is NOT the IK on clean EEF. Suspect model output or chunk stitching.")

    # --- plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        show = [0, 3, 6, 7, 10, 13]  # a few representative arm joints
        fig, axes = plt.subplots(len(show), 1, figsize=(12, 2.0*len(show)), sharex=True)
        tt = np.arange(T) / FPS
        for ax, j in zip(axes, show):
            ax.plot(tt, np.degrees(gt_arm[:, j]), lw=1.6, label="GT joint")
            ax.plot(tt, np.degrees(rec[:, j]), lw=1.0, ls="--", label="IK-recovered")
            ax.set_ylabel(f"j{j} (deg)"); ax.grid(alpha=0.3)
        axes[0].legend(loc="upper right", ncol=2); axes[-1].set_xlabel("time (s)")
        fig.suptitle(f"IK round-trip on GT EEF — episode {args.episode} "
                     f"(acc {ratio:.1f}x GT, {ik_tw} twitch frames)")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
        print(f"\nplot -> {args.plot}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
