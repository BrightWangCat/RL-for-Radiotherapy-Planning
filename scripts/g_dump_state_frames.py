# scripts/g_dump_state_frames.py
from __future__ import annotations

import os
import argparse
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# Ensure envs are registered
import rlfplan.register_envs  # noqa: F401


def _as_uint8_obs(obs: np.ndarray) -> np.ndarray:
    """Ensure obs is uint8 HWC with C=2."""
    if obs.dtype != np.uint8:
        # safe cast if obs is already in [0,255]
        obs = np.clip(obs, 0, 255).astype(np.uint8)
    if obs.shape != (96, 96, 2):
        raise ValueError(f"Expected obs shape (96,96,2), got {obs.shape}")
    return obs


def _contrast_limits_percentile(img_u8: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> Tuple[float, float]:
    """Compute display limits for better contrast without changing raw data."""
    vmin, vmax = np.percentile(img_u8.astype(np.float32), [p_low, p_high])
    # avoid degenerate ranges
    if vmax <= vmin + 1e-6:
        vmin, vmax = 0.0, 255.0
    return float(vmin), float(vmax)


def _format_meta_line(meta: Dict) -> str:
    """Build a concise metadata string for titles."""
    cid = meta.get("case_id", "")
    cp = meta.get("cp_idx", "")
    th = meta.get("theta_deg", "")
    d0 = meta.get("d0", "")
    x1 = meta.get("x1_mm", "")
    x2 = meta.get("x2_mm", "")
    r = meta.get("reward", None)
    if r is None:
        return f"case={cid}  cp={cp}  theta={th}°  d0={d0}  x1={x1}  x2={x2}"
    return f"case={cid}  cp={cp}  theta={th}°  d0={d0}  x1={x1}  x2={x2}  r={r:.4f}"


def save_state_frames(
    obs_u8: np.ndarray,
    meta: Dict,
    outdir: str,
    tag: str,
    stretch_frame1: bool = True,
    p_low: float = 1.0,
    p_high: float = 99.0,
    save_npz: bool = False,
):
    """
    Save:
      - {tag}_frame1.png
      - {tag}_frame2.png
      - {tag}_panel.png
      - optionally {tag}.npz (obs + meta)
    """
    os.makedirs(outdir, exist_ok=True)

    obs_u8 = _as_uint8_obs(obs_u8)
    f1 = obs_u8[:, :, 0]
    f2 = obs_u8[:, :, 1]

    # Print min/max for debugging and reporting
    print(f"[{tag}] frame1 min/max: {int(f1.min())} {int(f1.max())}")
    print(f"[{tag}] frame2 min/max: {int(f2.min())} {int(f2.max())}")
    print(f"[{tag}] meta: {_format_meta_line(meta)}")

    # Raw frame pngs (keep full 0..255)
    plt.imsave(os.path.join(outdir, f"{tag}_frame1.png"), f1, cmap="gray", vmin=0, vmax=255)
    plt.imsave(os.path.join(outdir, f"{tag}_frame2.png"), f2, cmap="gray", vmin=0, vmax=255)

    # Panel figure (better for slides)
    fig = plt.figure(figsize=(12, 4.6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    if stretch_frame1:
        vmin1, vmax1 = _contrast_limits_percentile(f1, p_low=p_low, p_high=p_high)
        ax1.imshow(f1, cmap="gray", vmin=vmin1, vmax=vmax1)
        ax1.set_title("State Frame 1: Dose - Objective (display stretched)")
    else:
        ax1.imshow(f1, cmap="gray", vmin=0, vmax=255)
        ax1.set_title("State Frame 1: Dose - Objective (raw uint8)")

    ax2.imshow(f2, cmap="gray", vmin=0, vmax=255)
    ax2.set_title("State Frame 2: Machine Sinogram (raw uint8)")

    # Metadata subtitle
    fig.suptitle(_format_meta_line(meta), y=0.98, fontsize=11)

    for ax in (ax1, ax2):
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(outdir, f"{tag}_panel.png"), dpi=220)
    plt.close(fig)

    # Optional: save data for reproducibility
    if save_npz:
        # store obs and meta dict as npz; meta becomes a 0-d object array
        np.savez_compressed(
            os.path.join(outdir, f"{tag}.npz"),
            obs=obs_u8,
            meta=np.array([meta], dtype=object),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", type=str, default="OpenKBPVMAT2D-v0")
    ap.add_argument("--case-id", type=str, default="", help="e.g., pt_241; empty uses env default")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--policy", type=str, default="fixed", choices=["fixed", "random"])
    ap.add_argument("--fixed-action", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=0, help="steps after reset before dumping sN")

    ap.add_argument("--outdir", type=str, default="state_frames")
    ap.add_argument("--tag", type=str, default="s")

    ap.add_argument("--stretch-frame1", action="store_true", default=True)
    ap.add_argument("--no-stretch-frame1", action="store_true", default=False)
    ap.add_argument("--p-low", type=float, default=1.0)
    ap.add_argument("--p-high", type=float, default=99.0)

    ap.add_argument("--save-npz", action="store_true", default=False)

    args = ap.parse_args()
    if args.no_stretch_frame1:
        args.stretch_frame1 = False

    # Construct env (your setup may require OPENKBP_CASE exported; sh handles it)
    env = gym.make(args.env_id)

    # Reset options
    options = {"case_id": args.case_id} if args.case_id else None
    obs, info = env.reset(seed=args.seed, options=options)
    obs = _as_uint8_obs(obs)

    # Meta at s0
    meta0 = dict(info) if isinstance(info, dict) else {}
    meta0.setdefault("case_id", args.case_id or meta0.get("case_id", ""))
    meta0.setdefault("cp_idx", meta0.get("cp_idx", 0))
    meta0.setdefault("theta_deg", meta0.get("theta_deg", 0.0))
    meta0["reward"] = None

    save_state_frames(
        obs_u8=obs,
        meta=meta0,
        outdir=args.outdir,
        tag=f"{args.tag}0",
        stretch_frame1=args.stretch_frame1,
        p_low=args.p_low,
        p_high=args.p_high,
        save_npz=args.save_npz,
    )

    # Step forward
    last_info: Dict = {}
    last_reward: float = 0.0
    for t in range(1, args.n_steps + 1):
        if args.policy == "random":
            a = env.action_space.sample()
        else:
            a = int(args.fixed_action)

        obs, r, term, trunc, step_info = env.step(a)
        obs = _as_uint8_obs(obs)
        last_reward = float(r)
        last_info = dict(step_info) if isinstance(step_info, dict) else {}

        if term or trunc:
            print(f"[warn] episode ended early at step {t}: term={term} trunc={trunc}")
            break

    # Meta at sN
    metaN = dict(last_info) if last_info else dict(meta0)
    metaN.setdefault("case_id", args.case_id or metaN.get("case_id", ""))
    metaN.setdefault("cp_idx", metaN.get("cp_idx", args.n_steps))
    metaN.setdefault("theta_deg", metaN.get("theta_deg", 3.75 * args.n_steps))
    metaN["reward"] = last_reward

    save_state_frames(
        obs_u8=obs,
        meta=metaN,
        outdir=args.outdir,
        tag=f"{args.tag}{args.n_steps}",
        stretch_frame1=args.stretch_frame1,
        p_low=args.p_low,
        p_high=args.p_high,
        save_npz=args.save_npz,
    )

    print("Saved panels:")
    print(os.path.join(args.outdir, f"{args.tag}0_panel.png"))
    print(os.path.join(args.outdir, f"{args.tag}{args.n_steps}_panel.png"))
    env.close()


if __name__ == "__main__":
    main()
