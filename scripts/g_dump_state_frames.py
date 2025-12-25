import os
import argparse
import numpy as np

import gymnasium as gym
import matplotlib.pyplot as plt

# make sure envs are registered
import rlfplan.register_envs  # noqa: F401


def save_frames(obs_uint8: np.ndarray, outdir: str, tag: str):
    assert obs_uint8.shape == (96, 96, 2), f"unexpected obs shape: {obs_uint8.shape}"
    f1 = obs_uint8[:, :, 0]
    f2 = obs_uint8[:, :, 1]

    os.makedirs(outdir, exist_ok=True)

    # save raw frames
    plt.imsave(os.path.join(outdir, f"{tag}_frame1.png"), f1, cmap="gray", vmin=0, vmax=255)
    plt.imsave(os.path.join(outdir, f"{tag}_frame2.png"), f2, cmap="gray", vmin=0, vmax=255)

    # save a panel (for slides)
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.imshow(f1, cmap="gray", vmin=0, vmax=255)
    ax1.set_title("State Frame 1: Dose - Objective (uint8)")
    ax1.axis("off")

    ax2.imshow(f2, cmap="gray", vmin=0, vmax=255)
    ax2.set_title("State Frame 2: Machine Sinogram (uint8)")
    ax2.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{tag}_panel.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", type=str, default="OpenKBPVMAT2D-v0")
    ap.add_argument("--case-id", type=str, default="", help="e.g., pt_241; empty uses env default")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--policy", type=str, default="fixed", choices=["fixed", "random"])
    ap.add_argument("--fixed-action", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=0, help="how many steps after reset before dumping frames")

    ap.add_argument("--outdir", type=str, default="state_frames")
    ap.add_argument("--tag", type=str, default="s")
    args = ap.parse_args()

    # env construction requires OPENKBP_CASE in some setups; keep your current practice
    env = gym.make(args.env_id)

    options = {"case_id": args.case_id} if args.case_id else None
    obs, info = env.reset(seed=args.seed, options=options)

    # dump s0
    save_frames(obs, args.outdir, f"{args.tag}0")

    # advance
    for t in range(1, args.n_steps + 1):
        if args.policy == "random":
            a = env.action_space.sample()
        else:
            a = int(args.fixed_action)
        obs, r, term, trunc, step_info = env.step(a)
        if term or trunc:
            break

    # dump s_{n_steps}
    save_frames(obs, args.outdir, f"{args.tag}{args.n_steps}")

    print("Saved:")
    print(os.path.join(args.outdir, f"{args.tag}0_panel.png"))
    print(os.path.join(args.outdir, f"{args.tag}{args.n_steps}_panel.png"))
    env.close()


if __name__ == "__main__":
    main()
