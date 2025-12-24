import os
import glob
import argparse
import numpy as np
import torch
import gymnasium as gym

PROJECT_ROOT = "/fs/scratch/PCON0023/mingshiw/RLfPlan5"
import sys
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "cleanrl"))

import rlfplan.register_envs  # noqa: F401
from cleanrl.ppo_discrete_cnn_vmat2d import Agent


def make_env(env_id: str):
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def find_latest_checkpoint(run_dir: str) -> str:
    pats = ["*.cleanrl_model", "*.pt", "*.pth"]
    cands = []
    for p in pats:
        cands.extend(glob.glob(os.path.join(run_dir, p)))
        cands.extend(glob.glob(os.path.join(run_dir, "**", p), recursive=True))
    cands = [c for c in cands if os.path.isfile(c)]
    if not cands:
        raise FileNotFoundError(f"No checkpoint found under {run_dir}")
    cands.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return cands[0]


@torch.no_grad()
def rollout(agent, env, device, max_steps: int, mode: str, seed: int):
    obs, info = env.reset(seed=seed)
    ret = 0.0
    err_sum = 0.0
    oar_sum = 0.0
    steps = 0
    last_info = {}

    for _ in range(max_steps):
        if mode == "random":
            a = env.action_space.sample()
        else:
            x = torch.tensor(obs, dtype=torch.uint8, device=device).unsqueeze(0)  # (1,96,96,2)
            logits, _ = agent.forward(x)
            if mode == "stochastic":
                dist = torch.distributions.Categorical(logits=logits)
                a = int(dist.sample().item())
            elif mode == "deterministic":
                a = int(torch.argmax(logits, dim=-1).item())
            else:
                raise ValueError(mode)

        obs, r, term, trunc, step_info = env.step(a)
        ret += float(r)
        err_sum += float(step_info.get("err_norm", 0.0))
        oar_sum += float(step_info.get("oar_pen_norm", 0.0))
        steps += 1
        last_info = step_info
        if term or trunc:
            break

    return {
        "return": ret,
        "mean_err_norm": err_sum / max(1, steps),
        "mean_oar_norm": oar_sum / max(1, steps),
        "steps": steps,
        "ptv70_mean": float(last_info.get("ptv70_mean", np.nan)),
        "ptv70_ref_mean": float(last_info.get("ptv70_ref_mean", np.nan)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--env-id", default="OpenKBPVMAT2D-v0")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=int(os.environ.get("OPENKBP_MAX_STEPS", "192")))
    ap.add_argument("--seed", type=int, default=0)

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--stochastic", action="store_true")
    g.add_argument("--random", action="store_true")
    args = ap.parse_args()

    mode = "deterministic"
    if args.stochastic:
        mode = "stochastic"
    if args.random:
        mode = "random"

    run_dir = args.run_dir.rstrip("/")
    ckpt_path = find_latest_checkpoint(run_dir)
    print("Using checkpoint:", ckpt_path)
    print("Eval mode:", mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build agent
    agent = Agent(n_actions=15).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(ckpt, strict=True)
    agent.eval()

    env = make_env(args.env_id)

    results = []
    for i in range(args.episodes):
        res = rollout(agent, env, device, args.max_steps, mode, seed=args.seed + i)
        results.append(res)
        print(
            f"[ep {i+1:02d}] return={res['return']:.4f} "
            f"mean_err_norm={res['mean_err_norm']:.4f} mean_oar_norm={res['mean_oar_norm']:.4f} "
            f"ptv70_mean={res['ptv70_mean']:.3f} ref={res['ptv70_ref_mean']:.3f}"
        )

    def mean(k): return float(np.mean([r[k] for r in results]))
    print("\n=== SUMMARY ===")
    print("episodes:", args.episodes)
    print("avg_return:", mean("return"))
    print("avg_mean_err_norm:", mean("mean_err_norm"))
    print("avg_mean_oar_norm:", mean("mean_oar_norm"))
    print("avg_ptv70_mean:", mean("ptv70_mean"))
    print("avg_ptv70_ref_mean:", mean("ptv70_ref_mean"))


if __name__ == "__main__":
    main()
