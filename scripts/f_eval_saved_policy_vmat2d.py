# scripts/f_eval_saved_policy_vmat2d.py
import os
import glob
import argparse
from typing import Optional

import numpy as np
import torch
import gymnasium as gym

PROJECT_ROOT = "/fs/scratch/PCON0023/mingshiw/RLfPlan5"
import sys
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "cleanrl"))

import rlfplan.register_envs  # noqa: F401
from rlfplan.wrappers.case_sampler import load_case_list


def pick_checkpoint(run_dir: str) -> str:
    p_best = os.path.join(run_dir, "model_best.cleanrl_model")
    if os.path.isfile(p_best):
        return p_best
    p_last = os.path.join(run_dir, "model_last.cleanrl_model")
    if os.path.isfile(p_last):
        return p_last
    p_compat = os.path.join(run_dir, "model.cleanrl_model")
    if os.path.isfile(p_compat):
        return p_compat

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


def safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class Agent(torch.nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(2, 32, 8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros((1, 2, 96, 96), dtype=torch.float32)
            n_flat = self.network(dummy).shape[1]
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(n_flat, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_actions),
        )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()
        return x.float() / 255.0

    def forward_logits(self, obs_uint8: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(obs_uint8)
        h = self.network(x)
        return self.actor(h)


@torch.no_grad()
def rollout(
    env: gym.Env,
    agent: Optional[Agent],
    device: torch.device,
    mode: str,
    max_steps: int,
    seed: int,
    case_id: Optional[str] = None,
    fixed_action: Optional[int] = None,
):
    options = {"case_id": case_id} if case_id else None
    obs, info = env.reset(seed=seed, options=options)

    ret = 0.0
    err_sum = 0.0
    oar_sum = 0.0
    steps = 0
    last_info = {}

    for _ in range(max_steps):
        if fixed_action is not None:
            a = int(fixed_action)
        elif mode == "random":
            a = env.action_space.sample()
        else:
            assert agent is not None
            x = torch.tensor(obs, dtype=torch.uint8, device=device).unsqueeze(0)
            logits = agent.forward_logits(x)
            dist = torch.distributions.Categorical(logits=logits)
            if mode == "stochastic":
                a = int(dist.sample().item())
            elif mode == "deterministic":
                a = int(torch.argmax(logits, dim=1).item())
            else:
                raise ValueError(f"unknown mode: {mode}")

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
        "case_id": case_id or str(info.get("case_id", "")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--env-id", type=str, default="OpenKBPVMAT2D-v0")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--fixed-action", type=int, default=None, help="If set, always take this action id.")

    ap.add_argument("--cases-file", type=str, default="")
    ap.add_argument("--episodes-per-case", type=int, default=1)

    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    # Mode selection
    mode = "deterministic"
    if args.random:
        mode = "random"
    if args.stochastic:
        mode = "stochastic"
    if args.fixed_action is not None:
        mode = f"fixed({args.fixed_action})"

    device = torch.device("cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda")

    # Case list
    if args.cases_file:
        case_ids = load_case_list(args.cases_file)
        if not os.environ.get("OPENKBP_CASE"):
            os.environ["OPENKBP_CASE"] = str(case_ids[0])
    else:
        case_ids = []

    env = gym.make(args.env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Agent only needed for deterministic/stochastic
    agent = None
    ckpt = None
    if args.fixed_action is None and mode != "random":
        ckpt = pick_checkpoint(args.run_dir)
        print(f"Using checkpoint: {ckpt}")
        n_actions = env.action_space.n
        agent = Agent(n_actions=n_actions).to(device)
        sd = safe_torch_load(ckpt, device)
        missing, unexpected = agent.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"[load_state_dict] ignored unexpected keys (showing up to 8): {unexpected[:8]}")
        if missing:
            print(f"[load_state_dict] missing keys (showing up to 8): {missing[:8]}")
        agent.eval()
    else:
        print("Using checkpoint: (none)")

    print(f"Eval mode: {mode}")

    metrics = []
    if case_ids:
        for cid in case_ids:
            for i in range(args.episodes_per_case):
                metrics.append(
                    rollout(
                        env, agent, device, "deterministic" if agent else "random",
                        args.max_steps, args.seed + 1000 * i,
                        case_id=cid, fixed_action=args.fixed_action
                    )
                )
    else:
        for i in range(args.episodes):
            metrics.append(
                rollout(
                    env, agent, device, "deterministic" if agent else "random",
                    args.max_steps, args.seed + 1000 * i,
                    case_id=None, fixed_action=args.fixed_action
                )
            )

    env.close()

    rets = [m["return"] for m in metrics]
    err = [m["mean_err_norm"] for m in metrics]
    oar = [m["mean_oar_norm"] for m in metrics]
    ptv = [m["ptv70_mean"] for m in metrics if np.isfinite(m["ptv70_mean"])]
    ptvref = [m["ptv70_ref_mean"] for m in metrics if np.isfinite(m["ptv70_ref_mean"])]

    print("=== SUMMARY ===")
    print(f"episodes: {len(metrics)}")
    print(f"avg_return: {float(np.mean(rets))}")
    print(f"avg_mean_err_norm: {float(np.mean(err))}")
    print(f"avg_mean_oar_norm: {float(np.mean(oar))}")
    if ptv:
        print(f"avg_ptv70_mean: {float(np.mean(ptv))}")
    if ptvref:
        print(f"avg_ptv70_ref_mean: {float(np.mean(ptvref))}")


if __name__ == "__main__":
    main()
