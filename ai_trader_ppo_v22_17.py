# ai_trader_ppo_v22_17.py
#
# V22_17: AI Trader PPO trainer (strong trend window version)
# - Keep rollout=8192, total_steps default 150k, entropy_coef=0.15
# - Add symbol -> days mapping:
#     BTCUSDT -> 500
#     ETHUSDT -> 540
#     BNBUSDT -> 500
#     SOLUSDT -> 360
#   If --days is given and > 0, it overrides the mapping.

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

from trading_env_v22_17 import TradingEnvV22MultiTFV17

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VERSION_TAG = "V22_17"
MODEL_DIR = "models"


class ActorCritic(nn.Module):
    """Simple MLP Actor-Critic for discrete PPO."""

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
    batch_size: int = 8192
    minibatch_size: int = 256
    entropy_coef: float = 0.15
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    valid_eval_interval: int = 1
    early_stop_patience: int = 8


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
    """Collect one batch of trajectories from train env."""
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


def evaluate_policy_on_env(policy, env, max_steps: int = 1000):
    """Evaluate policy on env using greedy (argmax prob) actions."""
    policy.eval()
    state = env.reset()
    done = False
    step_count = 0
    total_reward = 0.0
    last_info = {}

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
        last_info = info

    final_equity = last_info.get("equity", 0.0) if last_info else 0.0
    return final_equity, total_reward, step_count


def train_ppo(
    symbol: str,
    days: int,
    total_steps: int,
    config: PPOConfig,
    save_prefix: str,
) -> str:
    """Train PPO model and return path to best valid checkpoint."""
    env_train = TradingEnvV22MultiTFV17(symbol=symbol, days=days, segment="train")
    env_valid = TradingEnvV22MultiTFV17(symbol=symbol, days=days, segment="valid")
    env_test = TradingEnvV22MultiTFV17(symbol=symbol, days=days, segment="test")

    state_dim = env_train.state_dim
    action_dim = env_train.action_space_n

    logger.info(f"[PPO {VERSION_TAG}] config: symbol={symbol}, days={days}, total_steps={total_steps}")
    logger.info(f"[PPO {VERSION_TAG}] state_dim={state_dim}, action_dim={action_dim}, device={DEVICE}")

    policy = ActorCritic(state_dim=state_dim, action_dim=action_dim).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=config.lr)

    steps_collected = 0
    train_round = 0
    start_time = time.time()

    best_valid_equity = -1e18
    best_valid_path = os.path.join(MODEL_DIR, f"{save_prefix}_best_valid.pt")

    while steps_collected < total_steps:
        train_round += 1

        (
            states_np,
            actions_np,
            log_probs_old_np,
            rewards_np,
            dones_np,
            values_np,
        ) = collect_trajectories(env_train, policy, config.batch_size)

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
            f"[PPO {VERSION_TAG}] Round={train_round}, steps_collected={steps_collected}/{total_steps}, "
            f"batch_steps={steps_this_iter}, entropy_coef_now={entropy_coef_now:.5f}, elapsed={elapsed:.1f}s"
        )

        if train_round % config.valid_eval_interval == 0:
            valid_equity, valid_reward, valid_steps = evaluate_policy_on_env(
                policy, env_valid, max_steps=1000
            )
            logger.info(
                f"[VALID {VERSION_TAG}] Round={train_round}, equity={valid_equity:.2f}, "
                f"total_reward={valid_reward:.4f}, steps={valid_steps}"
            )

            if valid_equity > best_valid_equity:
                best_valid_equity = valid_equity
                torch.save(
                    {
                        "symbol": symbol,
                        "days": days,
                        "state_dim": state_dim,
                        "action_dim": action_dim,
                        "model_state_dict": policy.state_dict(),
                        "config": config.__dict__,
                    },
                    best_valid_path,
                )
                logger.info(f"[VALID {VERSION_TAG}] new best valid model saved: equity={best_valid_equity:.2f}")
            else:
                logger.info(f"[VALID {VERSION_TAG}] no improvement, continue training")

    if os.path.exists(best_valid_path):
        checkpoint = torch.load(best_valid_path, map_location=DEVICE)
        policy.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"[PPO {VERSION_TAG}] loaded best valid model: {best_valid_path}")
    else:
        logger.warning(f"[PPO {VERSION_TAG}] best_valid model not found, using current weights")

    env_test = TradingEnvV22MultiTFV17(symbol=symbol, days=days, segment="test")
    test_equity, test_reward, test_steps = evaluate_policy_on_env(policy, env_test, max_steps=1000)
    logger.info(
        f"[TEST {VERSION_TAG}] test result: equity={test_equity:.2f}, total_reward={test_reward:.4f}, steps={test_steps}"
    )

    return best_valid_path


def evaluate_model(
    model_path: str,
    symbol: str,
    days: int,
    eval_split: str = "test",
    max_steps: int = 1000,
) -> None:
    assert eval_split in ("train", "valid", "test"), f"Invalid eval_split: {eval_split}"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dim = checkpoint["state_dim"]
    action_dim = checkpoint["action_dim"]

    policy = ActorCritic(state_dim=state_dim, action_dim=action_dim).to(DEVICE)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    env = TradingEnvV22MultiTFV17(symbol=symbol, days=days, segment=eval_split)
    logger.info(f"[EVAL {VERSION_TAG}] evaluating model: {model_path} | split={eval_split}")

    final_equity, total_reward, steps = evaluate_policy_on_env(policy, env, max_steps=max_steps)
    logger.info(
        f"[EVAL {VERSION_TAG}] done: split={eval_split}, steps={steps}, "
        f"total_reward={total_reward:.4f}, final_equity={final_equity:.2f}"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description=f"AI Trader PPO Trainer ({VERSION_TAG})"
    )
    p.add_argument("--symbol", type=str, default="BTCUSDT", help="symbol, e.g., BTCUSDT")
    p.add_argument(
        "--days",
        type=int,
        default=0,
        help="use last N days of data; if 0, choose automatically by symbol (BTC=500, ETH=540, BNB=500, SOL=360, others=365)",
    )
    p.add_argument(
        "--total-steps",
        type=int,
        default=150000,
        help="total steps to sample during training",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="mode: train or eval",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default="",
        help="model path for eval mode; if empty, use default best_valid path.",
    )
    p.add_argument(
        "--eval-split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="which data split to use in eval mode",
    )
    return p.parse_args()


def auto_days_for_symbol(symbol: str) -> int:
    s = symbol.upper()
    if s == "BTCUSDT":
        return 500
    if s == "ETHUSDT":
        return 540
    if s == "BNBUSDT":
        return 500
    if s == "SOLUSDT":
        return 360
    return 365


def main():
    args = parse_args()
    symbol = args.symbol.upper()

    days = args.days if args.days and args.days > 0 else auto_days_for_symbol(symbol)

    ensure_model_dir()
    save_prefix = f"ai_trader_ppo_{VERSION_TAG}_{symbol}"

    logger.info(f"[MAIN {VERSION_TAG}] symbol={symbol}, days={days}, mode={args.mode}")

    if args.mode == "train":
        config = PPOConfig()
        best_valid_path = train_ppo(
            symbol=symbol,
            days=days,
            total_steps=args.total_steps,
            config=config,
            save_prefix=save_prefix,
        )
        logger.info(f"[MAIN {VERSION_TAG}] training done, best_valid_path={best_valid_path}")
    else:
        default_model_path = os.path.join(MODEL_DIR, f"ai_trader_ppo_{VERSION_TAG}_{symbol}_best_valid.pt")
        model_path = args.model_path or default_model_path
        evaluate_model(
            model_path=model_path,
            symbol=symbol,
            days=days,
            eval_split=args.eval_split,
            max_steps=1000,
        )


if __name__ == "__main__":
    main()
