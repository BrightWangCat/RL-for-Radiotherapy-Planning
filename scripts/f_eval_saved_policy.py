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


def make_env():
    """
    Reproduce the common CleanRL wrappers used in ppo_continuous_action:
      - RecordEpisodeStatistics
      - ClipAction
      - NormalizeObservation + clip
      - NormalizeReward + clip
    """
    env = gym.make("OpenKBPGrouped-v0")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    return env


def find_latest_checkpoint(run_dir: str) -> str:
    pats = ["*.cleanrl_model", "*.pt", "*.pth", "*model*"]
    cands = []
    for p in pats:
        cands.extend(glob.glob(os.path.join(run_dir, p)))
        cands.extend(glob.glob(os.path.join(run_dir, "**", p), recursive=True))
    cands = [c for c in cands if os.path.isfile(c)]
    if not cands:
        raise FileNotFoundError(f"No checkpoint found under {run_dir}")
    cands.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return cands[0]


def torch_load_weights(path: str, device: torch.device):
    # Avoid the torch FutureWarning when possible (your file is weights only)
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_agent(envs, device: torch.device):
    import cleanrl.ppo_continuous_action as ppo_mod
    agent = ppo_mod.Agent(envs).to(device)
    return agent


def load_state_dict_strict(agent: torch.nn.Module, ckpt_obj):
    if not isinstance(ckpt_obj, dict):
        raise ValueError(f"Checkpoint is not a dict/state_dict: {type(ckpt_obj)}")
    # strict=True：不允许静默失败
    agent.load_state_dict(ckpt_obj, strict=True)


@torch.no_grad()
def rollout(agent, env, device: torch.device, max_steps: int, deterministic: bool = True):
    obs, info = env.reset()
    ret = 0.0
    err_sum = 0.0
    oar_sum = 0.0
    steps = 0
    last_info = {}

    for _ in range(max_steps):
        x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        if deterministic:
            # Deterministic action: use actor mean (env has ClipAction wrapper)
            if not hasattr(agent, "actor_mean"):
                raise AttributeError("Agent has no attribute 'actor_mean' (unexpected CleanRL version).")
            action = agent.actor_mean(x)
        else:
            action, _, _, _ = agent.get_action_and_value(x)

        a = action.squeeze(0).detach().cpu().numpy()

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
    ap.add_argument("--max-steps", type=int, default=int(os.environ.get("OPENKBP_MAX_STEPS", "50")))
    ap.add_argument("--stochastic", action="store_true", help="use stochastic actions instead of deterministic mean")
    args = ap.parse_args()

    run_dir = args.run_dir.rstrip("/")
    ckpt_path = find_latest_checkpoint(run_dir)
    print("Using checkpoint:", ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build vector env for Agent signature, and a single env for rollout
    envs = gym.vector.SyncVectorEnv([lambda: make_env()])
    env = make_env()

    agent = build_agent(envs, device)
    ckpt = torch_load_weights(ckpt_path, device)
    load_state_dict_strict(agent, ckpt)
    agent.eval()



    results = []
    for i in range(args.episodes):
        res = rollout(agent, env, device, args.max_steps, deterministic=(not args.stochastic))
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
