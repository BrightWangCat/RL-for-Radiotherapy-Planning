# scripts/g_dump_state_frames_paperstyle.py
from __future__ import annotations

import os
import argparse
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

import rlfplan.register_envs  # noqa: F401


def _as_uint8_obs(obs: np.ndarray) -> np.ndarray:
    if obs.dtype != np.uint8:
        obs = np.clip(obs, 0, 255).astype(np.uint8)
    if obs.shape != (96, 96, 2):
        raise ValueError(f"Expected obs shape (96,96,2), got {obs.shape}")
    return obs


def _scale_to_bin(v: float, vmin: float, vmax: float, width: int) -> int:
    if width <= 1:
        return 0
    if vmax <= vmin:
        return 0
    x = (v - vmin) / (vmax - vmin)
    x = float(np.clip(x, 0.0, 1.0))
    return int(round(x * (width - 1)))


def build_paper_sinogram(
    records: List[Dict],
    d0_min: float,
    d0_max: float,
    x_min: float,
    x_max: float,
    d0_cols: int = 30,
) -> np.ndarray:
    """
    Paper-like encoding:
      - y-axis: relative theta / control point index (0..95)
      - x-axis: [d0 | x1 | x2] segments
      - intensity: d0=200, x1=150, x2=100 (background 0)
    """
    H, W = 96, 96
    frame = np.zeros((H, W), dtype=np.uint8)

    seg_d0 = d0_cols
    seg_x1 = (W - seg_d0) // 2
    seg_x2 = W - seg_d0 - seg_x1

    for rec in records:
        cp = int(rec.get("cp_idx", 0))
        if cp < 0:
            continue
        # in paper, one arc is 96 CPs; wrap into 0..95 for visualization
        y = cp % 96

        d0 = float(rec.get("d0", 0.0))
        x1 = float(rec.get("x1_mm", 0.0))
        x2 = float(rec.get("x2_mm", 0.0))

        xd0 = _scale_to_bin(d0, d0_min, d0_max, seg_d0)
        xx1 = _scale_to_bin(x1, x_min, x_max, seg_x1)
        xx2 = _scale_to_bin(x2, x_min, x_max, seg_x2)

        frame[y, xd0] = 200
        frame[y, seg_d0 + xx1] = 150
        frame[y, seg_d0 + seg_x1 + xx2] = 100

    return frame


def save_panel(frame1_u8: np.ndarray, sinogram_u8: np.ndarray, meta: str, outpath: str):
    fig = plt.figure(figsize=(12, 4.6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.imshow(frame1_u8, cmap="gray", vmin=0, vmax=255)
    ax1.set_title("State Frame 1 (env): Dose - Objective (uint8)")
    ax1.axis("off")

    ax2.imshow(sinogram_u8, cmap="gray", vmin=0, vmax=255)
    ax2.set_title("State Frame 2 (paper-style): sinogram (uint8)")
    ax2.axis("off")

    fig.suptitle(meta, y=0.98, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", type=str, default="OpenKBPVMAT2D-v0")
    ap.add_argument("--case-id", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--policy", type=str, default="fixed", choices=["fixed", "random"])
    ap.add_argument("--fixed-action", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=10)

    ap.add_argument("--outdir", type=str, default="state_frames_paperstyle")
    ap.add_argument("--tag", type=str, default="paper")

    # encoding ranges (paper uses real ranges; adjust if your env differs)
    ap.add_argument("--d0-min", type=float, default=20.0)
    ap.add_argument("--d0-max", type=float, default=600.0)
    ap.add_argument("--x-min", type=float, default=0.0)
    ap.add_argument("--x-max", type=float, default=240.0)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    env = gym.make(args.env_id)
    options = {"case_id": args.case_id} if args.case_id else None
    obs, info = env.reset(seed=args.seed, options=options)
    obs = _as_uint8_obs(obs)

    records: List[Dict] = []
    # record s0 machine params
    if isinstance(info, dict):
        records.append(dict(info))

    last_info: Dict = dict(info) if isinstance(info, dict) else {}
    last_r: float = 0.0

    for t in range(1, args.n_steps + 1):
        a = env.action_space.sample() if args.policy == "random" else int(args.fixed_action)
        obs, r, term, trunc, step_info = env.step(a)
        obs = _as_uint8_obs(obs)
        last_r = float(r)
        last_info = dict(step_info) if isinstance(step_info, dict) else {}
        records.append(dict(last_info))
        if term or trunc:
            break

    # Frame1 = env frame1 at sN (what the policy actually sees)
    frame1 = obs[:, :, 0]

    # Frame2 = paper-style sinogram built from trajectory records
    sino = build_paper_sinogram(
        records,
        d0_min=args.d0_min,
        d0_max=args.d0_max,
        x_min=args.x_min,
        x_max=args.x_max,
        d0_cols=30,
    )

    cid = args.case_id or last_info.get("case_id", "")
    cp = int(last_info.get("cp_idx", args.n_steps))
    th = float(last_info.get("theta_deg", 3.75 * cp))
    meta = f"case={cid}  cp={cp}  theta={th:.1f}°  policy={args.policy}  r_last={last_r:.4f}"

    outpath = os.path.join(args.outdir, f"{args.tag}_s{cp:03d}.png")
    save_panel(frame1, sino, meta, outpath)

    print("Saved:", outpath)
    env.close()


if __name__ == "__main__":
    main()
