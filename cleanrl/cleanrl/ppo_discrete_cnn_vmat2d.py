# cleanrl/cleanrl/ppo_discrete_cnn_vmat2d.py
from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import asdict
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from rlfplan.wrappers.case_sampler import CaseSamplerConfig, CaseSamplerWrapper, load_case_list


def parse_args():
    parser = argparse.ArgumentParser()

    # CleanRL standard-ish args
    parser.add_argument("--env-id", type=str, default="OpenKBPVMAT2D-v0")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--anneal-lr", action="store_true", default=False)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--clip-vloss", action="store_true", default=True)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--torch-deterministic", action="store_true", default=True)

    parser.add_argument("--capture-video", action="store_true", default=False)
    parser.add_argument("--save-model", action="store_true", default=True)
    parser.add_argument("--exp-name", type=str, default="ppo_discrete_cnn")

    # 3A multicase training knobs
    parser.add_argument("--train-cases-file", type=str, default="")
    parser.add_argument("--val-cases-file", type=str, default="")
    parser.add_argument("--case-sample-mode", type=str, default="random", choices=["random", "round_robin"])

    # paper-style periodic val selection
    parser.add_argument("--eval-every-updates", type=int, default=0,
                        help="If >0, run deterministic eval on val cases every N PPO updates and track best.")
    parser.add_argument("--eval-episodes-per-case", type=int, default=1)
    parser.add_argument("--eval-max-steps", type=int, default=0,
                        help="0 means use env's max_steps (recommended).")

    args = parser.parse_args()

    # derived sizes
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


def set_seed(seed: int, torch_deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = bool(torch_deterministic)


class Agent(nn.Module):
    def __init__(self, envs: gym.vector.VectorEnv):
        super().__init__()
        n_actions = envs.single_action_space.n

        # obs: HWC uint8 -> CHW float
        self.network = nn.Sequential(
            nn.Conv2d(2, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros((1, 2, 96, 96), dtype=torch.float32)
            n_flat = self.network(dummy).shape[1]

        self.actor = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C) uint8
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        return x.float() / 255.0

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(x)
        h = self.network(x)
        return self.critic(h)

    def get_action_and_value(self, x: torch.Tensor, action: Optional[torch.Tensor] = None):
        x = self._preprocess(x)
        h = self.network(x)
        logits = self.actor(h)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(h), logits


def make_env(env_id: str, idx: int, seed: int, capture_video: bool, run_name: str,
             train_cases: Optional[List[str]], case_sample_mode: str):
    def thunk():
        env = gym.make(env_id)

        # Per-episode multicase sampling (critical with VectorEnv autoreset)
        if train_cases:
            cfg = CaseSamplerConfig(case_ids=train_cases, mode=case_sample_mode, seed=seed + 1000 * idx)
            env = CaseSamplerWrapper(env, cfg)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        return env

    return thunk


@torch.no_grad()
def eval_on_cases(agent: Agent, env_id: str, device: torch.device,
                  case_ids: List[str], episodes_per_case: int,
                  max_steps: int, seed: int) -> Dict[str, float]:
    """
    Deterministic evaluation: argmax over logits.
    """
    if len(case_ids) == 0:
        raise ValueError("eval_on_cases got empty case_ids")

    # Ensure env can be constructed (some envs require OPENKBP_CASE at init-time)
    if not os.environ.get("OPENKBP_CASE"):
        os.environ["OPENKBP_CASE"] = str(case_ids[0])

    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    rets = []
    mean_err = []
    mean_oar = []

    for ci, cid in enumerate(case_ids):
        for ei in range(int(episodes_per_case)):
            obs, info = env.reset(seed=seed + 10_000 * ci + 100 * ei, options={"case_id": str(cid)})

            # If env exposes max_steps in reset info, use it when requested
            steps = 0
            err_sum = 0.0
            oar_sum = 0.0
            ret = 0.0

            while True:
                x = torch.tensor(obs, dtype=torch.uint8, device=device).unsqueeze(0)
                _, _, _, _, logits = agent.get_action_and_value(x)
                a = torch.argmax(logits, dim=1).item()

                obs, r, term, trunc, step_info = env.step(a)
                ret += float(r)
                err_sum += float(step_info.get("err_norm", 0.0))
                oar_sum += float(step_info.get("oar_pen_norm", 0.0))
                steps += 1

                if term or trunc:
                    break
                if max_steps > 0 and steps >= max_steps:
                    break

            rets.append(ret)
            mean_err.append(err_sum / max(1, steps))
            mean_oar.append(oar_sum / max(1, steps))

    env.close()
    return {
        "avg_return": float(np.mean(rets)) if rets else float("nan"),
        "avg_mean_err_norm": float(np.mean(mean_err)) if mean_err else float("nan"),
        "avg_mean_oar_norm": float(np.mean(mean_oar)) if mean_oar else float("nan"),
        "episodes": float(len(rets)),
    }


def main():
    args = parse_args()

    # Allow external env var seed override (keeps your existing workflow)
    if os.environ.get("OPENKBP_SEED") is not None:
        args.seed = int(os.environ["OPENKBP_SEED"])

    set_seed(args.seed, args.torch_deterministic)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Load case splits (3A)
    train_cases = load_case_list(args.train_cases_file) if args.train_cases_file else []
    val_cases = load_case_list(args.val_cases_file) if args.val_cases_file else []

    # Some envs require OPENKBP_CASE at init. Set a safe default.
    if not os.environ.get("OPENKBP_CASE"):
        if train_cases:
            os.environ["OPENKBP_CASE"] = str(train_cases[0])
        elif val_cases:
            os.environ["OPENKBP_CASE"] = str(val_cases[0])

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", str(vars(args)))

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.seed, args.capture_video, run_name, train_cases, args.case_sample_mode)
         for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "VMAT2D PPO expects Discrete action space"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, dtype=torch.uint8, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.int64, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)

    # Init
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.uint8, device=device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)

    num_updates = args.total_timesteps // args.batch_size

    best_val_return = -1e18
    best_path = None

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / float(num_updates)
            lrnow = frac * args.learning_rate
            for pg in optimizer.param_groups:
                pg["lr"] = lrnow

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, entropy, value, _ = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, term, trunc, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(term, trunc)

            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.uint8, device=device)
            next_done = torch.tensor(done, dtype=torch.float32, device=device)

            # logging episodic returns if available
            if "final_info" in infos:
                for fi in infos["final_info"]:
                    if fi and "episode" in fi:
                        writer.add_scalar("charts/episodic_return", fi["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", fi["episode"]["l"], global_step)

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues.flatten() * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # PPO update
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # logging
        y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        print(f"update {update:04d}/{num_updates} step={global_step} SPS={sps}")

        # Save last checkpoint
        if args.save_model:
            run_dir = f"runs/{run_name}"
            os.makedirs(run_dir, exist_ok=True)
            last_path = os.path.join(run_dir, "model_last.cleanrl_model")
            torch.save(agent.state_dict(), last_path)
            # also keep compat name
            compat_path = os.path.join(run_dir, "model.cleanrl_model")
            torch.save(agent.state_dict(), compat_path)

        # Periodic validation (paper-style best selection)
        if args.eval_every_updates > 0 and val_cases and (update % args.eval_every_updates == 0):
            agent.eval()
            max_steps = args.eval_max_steps if args.eval_max_steps > 0 else 0
            val = eval_on_cases(
                agent=agent,
                env_id=args.env_id,
                device=device,
                case_ids=val_cases,
                episodes_per_case=args.eval_episodes_per_case,
                max_steps=max_steps,
                seed=args.seed + 999,
            )
            agent.train()

            writer.add_scalar("val/avg_return", val["avg_return"], global_step)
            writer.add_scalar("val/avg_mean_err_norm", val["avg_mean_err_norm"], global_step)
            writer.add_scalar("val/avg_mean_oar_norm", val["avg_mean_oar_norm"], global_step)

            # Track best
            if val["avg_return"] > best_val_return:
                best_val_return = val["avg_return"]
                if args.save_model:
                    best_path = os.path.join(f"runs/{run_name}", "model_best.cleanrl_model")
                    torch.save(agent.state_dict(), best_path)
                    best_path_meta = os.path.join(f"runs/{run_name}", "best_val.txt")
                    with open(best_path_meta, "w", encoding="utf-8") as f:
                        f.write(f"best_val_return={best_val_return}\n")
                        f.write(f"val_metrics={val}\n")
                        f.write(f"args={vars(args)}\n")
                print(f"[VAL] new best avg_return={best_val_return:.6f} saved to {best_path}")

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
