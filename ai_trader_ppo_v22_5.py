"""ai_trader_ppo_v22_5.py - V22_5 PPO 训练器"""

from __future__ import annotations

import argparse
import os
import time
import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from trading_env_v22_5 import TradingEnvV22MultiTF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VERSION_TAG = "V22_5"
MODEL_DIR = "models"


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def act(self, state: np.ndarray) -> Tuple[int, float, float]:
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits, value = self.forward(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    train_epochs: int = 10
    batch_size: int = 2048
    minibatch_size: int = 256
    entropy_coef: float = 0.05
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


def ensure_model_dir() -> None:
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)


def compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        adv[t] = last_gae
    returns = adv + values[:-1]
    return adv, returns


def normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = x.mean()
    std = x.std()
    return (x - mean) / (std + eps)


def collect_trajectories(env, policy, batch_size):
    policy.eval()
    states = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    values = []

    state = env.reset()
    done = False

    while len(rewards) < batch_size:
        action, log_p, v = policy.act(state)
        next_state, reward, done, info = env.step(action)

        states.append(state)
        actions.append(action)
        log_probs.append(log_p)
        rewards.append(reward)
        dones.append(float(done))
        values.append(v)

        state = next_state
        if done:
            state = env.reset()
            done = False

    state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        _, last_value = policy.forward(state_t)
    values.append(float(last_value.item()))

    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.int64),
        np.array(log_probs, dtype=np.float32),
        np.array(rewards, dtype=np.float32),
        np.array(dones, dtype=np.float32),
        np.array(values, dtype=np.float32),
    )


def train_ppo(symbol, days, total_steps, config: PPOConfig, save_path: str):
    env = TradingEnvV22MultiTF(symbol=symbol, days=days)
    state_dim = env.state_dim
    action_dim = env.action_space_n

    logger.info(f"[PPO] 训练配置: symbol={symbol}, days={days}, total_steps={total_steps}")
    logger.info(f"[PPO] 状态维度={state_dim}, 动作维度={action_dim}, 设备={DEVICE}")

    policy = ActorCritic(state_dim=state_dim, action_dim=action_dim).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=config.lr)

    steps_collected = 0
    start_time = time.time()

    while steps_collected < total_steps:
        (
            states_np,
            actions_np,
            log_probs_old_np,
            rewards_np,
            dones_np,
            values_np,
        ) = collect_trajectories(env, policy, config.batch_size)

        steps_this_iter = len(rewards_np)
        steps_collected += steps_this_iter

        adv_np, returns_np = compute_gae(
            rewards_np, values_np, dones_np, config.gamma, config.lam
        )
        adv_np = normalize(adv_np)

        states = torch.tensor(states_np, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions_np, dtype=torch.int64, device=DEVICE)
        old_log_probs = torch.tensor(log_probs_old_np, dtype=torch.float32, device=DEVICE)
        returns_t = torch.tensor(returns_np, dtype=torch.float32, device=DEVICE)
        adv_t = torch.tensor(adv_np, dtype=torch.float32, device=DEVICE)

        policy.train()
        num_samples = len(states)
        idxs = np.arange(num_samples)

        progress = min(1.0, steps_collected / float(total_steps))
        entropy_coef_now = config.entropy_coef * (1.0 - 0.7 * progress) + 1e-4

        for epoch in range(config.train_epochs):
            np.random.shuffle(idxs)
            for start in range(0, num_samples, config.minibatch_size):
                end = start + config.minibatch_size
                mb_idx = idxs[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_adv = adv_t[mb_idx]

                new_log_probs, values_pred, entropy = policy.evaluate_actions(mb_states, mb_actions)

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (mb_returns - values_pred).pow(2).mean()
                loss = policy_loss + config.value_coef * value_loss - entropy_coef_now * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                optimizer.step()

        elapsed = time.time() - start_time
        logger.info(
            f"[PPO] 已采样步数: {steps_collected}/{total_steps}, 本轮样本数={steps_this_iter}, "
            f"entropy_coef_now={entropy_coef_now:.5f}, 用时={elapsed:.1f}s"
        )

    ensure_model_dir()
    torch.save(
        {
            "symbol": symbol,
            "days": days,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "model_state_dict": policy.state_dict(),
            "config": config.__dict__,
        },
        save_path,
    )
    logger.info(f"[PPO] 模型已保存: {save_path}")


def evaluate_model(model_path: str, symbol: str, days: int, max_steps: int = 1000):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dim = checkpoint["state_dim"]
    action_dim = checkpoint["action_dim"]

    policy = ActorCritic(state_dim=state_dim, action_dim=action_dim).to(DEVICE)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    env = TradingEnvV22MultiTF(symbol=symbol, days=days)
    state = env.reset()
    done = False
    step_count = 0
    total_reward = 0.0

    logger.info(f"[EVAL] 开始评估模型: {model_path}")
    logger.info(f"[EVAL] 初始 equity={env.equity:.2f}")

    while not done and step_count < max_steps:
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits, value = policy.forward(state_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.probs.argmax(dim=-1).item()

        next_state, reward, done, info = env.step(int(action))
        total_reward += reward
        step_count += 1
        state = next_state

        if step_count % 50 == 0 or done:
            logger.info(
                f"[EVAL] step={step_count}, action={action}, reward={reward:.6f}, equity={info['equity']:.2f}"
            )

    logger.info(
        f"[EVAL] 评估结束: 步数={step_count}, 总reward={total_reward:.4f}, 最终权益={info['equity']:.2f}"
    )


def parse_args():
    p = argparse.ArgumentParser(description=f"AI Trader PPO 训练器 ({VERSION_TAG})")
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--total-steps", type=int, default=100000)
    p.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    p.add_argument("--model-path", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    symbol = args.symbol
    days = args.days

    ensure_model_dir()
    default_model_path = os.path.join(MODEL_DIR, f"ai_trader_ppo_{VERSION_TAG}_{symbol}.pt")

    if args.mode == "train":
        config = PPOConfig()
        save_path = args.model_path or default_model_path
        train_ppo(symbol, days, args.total_steps, config, save_path)
        evaluate_model(save_path, symbol, days, max_steps=1000)
    else:
        model_path = args.model_path or default_model_path
        evaluate_model(model_path, symbol, days, max_steps=1000)


if __name__ == "__main__":
    main()
