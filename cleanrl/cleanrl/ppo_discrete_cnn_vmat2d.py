# cleanrl/cleanrl/ppo_discrete_cnn_vmat2d.py
from __future__ import annotations

import os
import time
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


@dataclass
class Args:
    env_id: str = "OpenKBPVMAT2D-v0"
    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    num_envs: int = 4
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 0
    cuda: bool = True
    torch_deterministic: bool = True
    save_model: bool = True
    run_name: str = ""


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default=Args.env_id)
    p.add_argument("--total-timesteps", type=int, default=Args.total_timesteps)
    p.add_argument("--learning-rate", type=float, default=Args.learning_rate)
    p.add_argument("--num-envs", type=int, default=Args.num_envs)
    p.add_argument("--num-steps", type=int, default=Args.num_steps)
    p.add_argument("--gamma", type=float, default=Args.gamma)
    p.add_argument("--gae-lambda", type=float, default=Args.gae_lambda)
    p.add_argument("--num-minibatches", type=int, default=Args.num_minibatches)
    p.add_argument("--update-epochs", type=int, default=Args.update_epochs)
    p.add_argument("--clip-coef", type=float, default=Args.clip_coef)
    p.add_argument("--ent-coef", type=float, default=Args.ent_coef)
    p.add_argument("--vf-coef", type=float, default=Args.vf_coef)
    p.add_argument("--max-grad-norm", type=float, default=Args.max_grad_norm)
    p.add_argument("--seed", type=int, default=Args.seed)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--torch-deterministic", action="store_true")
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--run-name", type=str, default="")
    a = p.parse_args()

    args = Args(
        env_id=a.env_id,
        total_timesteps=a.total_timesteps,
        learning_rate=a.learning_rate,
        num_envs=a.num_envs,
        num_steps=a.num_steps,
        gamma=a.gamma,
        gae_lambda=a.gae_lambda,
        num_minibatches=a.num_minibatches,
        update_epochs=a.update_epochs,
        clip_coef=a.clip_coef,
        ent_coef=a.ent_coef,
        vf_coef=a.vf_coef,
        max_grad_norm=a.max_grad_norm,
        seed=a.seed,
        cuda=bool(a.cuda),
        torch_deterministic=bool(a.torch_deterministic),
        save_model=bool(a.save_model),
        run_name=str(a.run_name),
    )
    return args


def make_env(env_id: str, idx: int, seed: int):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env
    return thunk


class Agent(nn.Module):
    """
    CNN (classic Atari-like conv stack):
      Conv1: 32 @ 8x8 stride4
      Conv2: 64 @ 4x4 stride2
      Conv3: 64 @ 3x3 stride1
      FC: 512
      Policy: 15 logits
      Value: 1

    Input: uint8 (N,96,96,2) -> float in [0,1], then (N,2,96,96)
    """
    def __init__(self, n_actions: int = 15):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros((1, 2, 96, 96), dtype=torch.float32)
            n_flat = int(self.cnn(dummy).view(1, -1).shape[1])

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 512),
            nn.ReLU(),
        )
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, obs_u8: torch.Tensor):
        # obs_u8: (N,96,96,2) uint8/float
        if obs_u8.dtype != torch.float32:
            obs = obs_u8.float()
        else:
            obs = obs_u8
        obs = obs / 255.0
        obs = obs.permute(0, 3, 1, 2)  # N,H,W,C -> N,C,H,W
        x = self.cnn(obs)
        x = self.fc(x)
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value

    def get_action_and_value(self, obs_u8: torch.Tensor, action: torch.Tensor | None = None):
        """
        IMPORTANT: No @torch.no_grad() here.
        - Rollout sampling should wrap this call in `with torch.no_grad():`
        - PPO update should call it normally so gradients flow through logits/value.
        """
        logits, value = self.forward(obs_u8)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logprob, entropy, value


def main():
    args = parse_args()

    # Ensure env registered
    import rlfplan.register_envs  # noqa: F401

    run_name = args.run_name or f"{args.env_id}__ppo_discrete_cnn__{args.seed}__{int(time.time())}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, i, args.seed) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)
    assert tuple(envs.single_observation_space.shape) == (96, 96, 2)

    agent = Agent(n_actions=envs.single_action_space.n).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    num_updates = args.total_timesteps // (args.num_envs * args.num_steps)
    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches

    obs = torch.zeros((args.num_steps, args.num_envs, 96, 96, 2), dtype=torch.uint8, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.int64, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.uint8, device=device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)

    global_step = 0
    start_time = time.time()

    for update in range(1, num_updates + 1):
        # rollout
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)

            actions[step] = action
            logprobs[step] = logprob
            values[step] = value

            next_obs_np, reward_np, term_np, trunc_np, _ = envs.step(action.cpu().numpy())
            done_np = np.logical_or(term_np, trunc_np)

            rewards[step] = torch.tensor(reward_np, dtype=torch.float32, device=device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.uint8, device=device)
            next_done = torch.tensor(done_np, dtype=torch.float32, device=device)

        # bootstrap value
        with torch.no_grad():
            _, next_value = agent.forward(next_obs)

        advantages = torch.zeros_like(rewards, device=device)
        lastgaelam = 0.0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values

        # flatten batch
        b_obs = obs.reshape((-1, 96, 96, 2))
        b_actions = actions.reshape((-1,))
        b_logprobs = logprobs.reshape((-1,))
        b_advantages = advantages.reshape((-1,))
        b_returns = returns.reshape((-1,))
        b_values = values.reshape((-1,))

        # normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # PPO update
        inds = np.arange(batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = inds[start:start + minibatch_size]

                # NO no_grad here (we need gradients)
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # policy loss
                pg_loss1 = -b_advantages[mb_inds] * ratio
                pg_loss2 = -b_advantages[mb_inds] * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                v_loss = 0.5 * (b_returns[mb_inds] - newvalue).pow(2).mean()

                # entropy
                ent_loss = entropy.mean()

                loss = pg_loss - args.ent_coef * ent_loss + args.vf_coef * v_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        sps = int(global_step / max(1e-6, (time.time() - start_time)))
        print(f"update {update:04d}/{num_updates} step={global_step} SPS={sps}")

        if args.save_model and (update == num_updates or update % 10 == 0):
            path = os.path.join(run_dir, "model.cleanrl_model")
            torch.save(agent.state_dict(), path)

    envs.close()
    print("run_dir:", run_dir)


if __name__ == "__main__":
    main()
