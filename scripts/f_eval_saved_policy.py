import os
import glob
import argparse
import numpy as np
import torch
import gymnasium as gym

# Ensure project + cleanrl import paths
PROJECT_ROOT = "/fs/scratch/PCON0023/mingshiw/RLfPlan5"
import sys
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "cleanrl"))

import rlfplan.register_envs  # noqa: F401  (registers OpenKBPGrouped-v0)


def find_latest_checkpoint(run_dir: str) -> str:
    pats = ["*.pt", "*.pth", "*model*"]
    cands = []
    for p in pats:
        cands.extend(glob.glob(os.path.join(run_dir, p)))
        cands.extend(glob.glob(os.path.join(run_dir, "**", p), recursive=True))
    cands = [c for c in cands if os.path.isfile(c)]
    if not cands:
        raise FileNotFoundError(f"No checkpoint found under {run_dir}")
    cands.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return cands[0]


def load_agent(envs, ckpt_path: str, device: torch.device):
    # Import Agent definition from CleanRL continuous PPO script
    import cleanrl.ppo_continuous_action as ppo_mod

    agent = ppo_mod.Agent(envs).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    # Robust load: support different checkpoint dict formats
    if isinstance(ckpt, dict):
        for key in ["agent", "state_dict", "model_state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                agent.load_state_dict(ckpt[key], strict=False)
                return agent
        # sometimes ckpt itself is a state_dict-like dict
        agent.load_state_dict(ckpt, strict=False)
        return agent
    else:
        # rare case: whole model saved
        try:
            agent.load_state_dict(ckpt, strict=False)
            return agent
        except Exception:
            raise ValueError(f"Unrecognized checkpoint format: {type(ckpt)}")


@torch.no_grad()
def rollout(agent, env, device, deterministic: bool, max_steps: int):
    obs, info = env.reset()
    ret = 0.0
    err_sum = 0.0
    oar_sum = 0.0
    steps = 0

    last_info = {}

    for _ in range(max_steps):
        x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        # CleanRL API: get_action_and_value returns action (and others)
        # Deterministic eval: we still call the method; many versions sample stochastically.
        # For stability now, keep stochastic but seed-controlled; deterministic flag is kept for future extension.
        action, _, _, _ = agent.get_action_and_value(x)
        a = action.squeeze(0).cpu().numpy()

        obs, r, term, trunc, step_info = env.step(a)
        ret += float(r)
        err_sum += float(step_info.get("err_norm", 0.0))
        oar_sum += float(step_info.get("oar_pen_norm", 0.0))
        steps += 1
        last_info = step_info

        if term or trunc:
            break

    out = {
        "return": ret,
        "mean_err_norm": err_sum / max(1, steps),
        "mean_oar_norm": oar_sum / max(1, steps),
        "steps": steps,
        "ptv70_mean": float(last_info.get("ptv70_mean", np.nan)),
        "ptv70_ref_mean": float(last_info.get("ptv70_ref_mean", np.nan)),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to a runs/<run_name> directory")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--max-steps", type=int, default=int(os.environ.get("OPENKBP_MAX_STEPS", "50")))
    args = ap.parse_args()

    run_dir = args.run_dir.rstrip("/")
    ckpt = find_latest_checkpoint(run_dir)
    print("Using checkpoint:", ckpt)

    # Single env for evaluation
    env = gym.make("OpenKBPGrouped-v0")
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Vector env wrapper to satisfy CleanRL Agent(envs) signature
    envs = gym.vector.SyncVectorEnv([lambda: gym.make("OpenKBPGrouped-v0")])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = load_agent(envs, ckpt, device)
    agent.eval()

    results = []
    for i in range(args.episodes):
        res = rollout(agent, env, device, args.deterministic, args.max_steps)
        results.append(res)
        print(
            f"[ep {i+1:02d}] return={res['return']:.4f} "
            f"mean_err_norm={res['mean_err_norm']:.4f} mean_oar_norm={res['mean_oar_norm']:.4f} "
            f"ptv70_mean={res['ptv70_mean']:.3f} ref={res['ptv70_ref_mean']:.3f}"
        )

    # summary
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
