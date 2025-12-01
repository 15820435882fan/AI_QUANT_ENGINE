"""
ai_trader_ppo_v22_3.py

V22_3: AI Trader · PPO 训练器（基于 trading_env_v22_2.TradingEnvV22MultiTF）

功能概述：
- 使用多周期强化学习环境 TradingEnvV22MultiTF（1h 主轴 + 4h/15m 特征）；
- 采用 on-policy PPO 算法训练一个离散动作策略网络（4 动作：不动 / 开多 / 开空 / 平仓）；
- 支持命令行参数配置训练步数；
- 支持保存 / 加载模型，并在训练后进行一次评估回放。

使用前提：
- 需确保同目录下存在：local_data_engine.py、trading_env_v22_2.py
- 需安装 PyTorch：
    pip install torch

基础用法（训练示例）：
    python ai_trader_ppo_v22_3.py --symbol BTCUSDT --days 365 --total-steps 20000

训练完成后，会在 ./models 目录生成类似：
    ai_trader_ppo_V22_3_BTCUSDT.pt

评估（加载已训练模型跑一条 episode）：
    python ai_trader_ppo_v22_3.py --symbol BTCUSDT --days 365 --mode eval --model-path models/ai_trader_ppo_V22_3_BTCUSDT.pt
"""

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

from trading_env_v22_4 import TradingEnvV22MultiTF

# ===== 日志配置 =====

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VERSION_TAG = "V22_3"
MODEL_DIR = "models"


# ===== Actor-Critic 网络结构 =====

class ActorCritic(nn.Module):
    """简单两层 MLP 的 Actor-Critic，用于 4 动作 PPO。"""

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
        """
        给定单个 state，采样一个动作，并返回:
            action, log_prob_old, value
        """
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
        """
        批量计算：
            - log_probs(new)
            - values
            - entropy
        """
        logits, values = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy


# ===== PPO 配置 =====

@dataclass
class PPOConfig:
    gamma: float = 0.99          # 折扣因子
    lam: float = 0.95            # GAE 衰减
    clip_ratio: float = 0.2      # PPO clip
    lr: float = 3e-4             # 学习率
    train_epochs: int = 10       # 每批数据迭代次数
    batch_size: int = 2048       # 每次采样步数
    minibatch_size: int = 256    # 每次更新子批大小
    entropy_coef: float = 0.01   # 熵奖励系数
    value_coef: float = 0.5      # 价值函数损失权重
    max_grad_norm: float = 0.5   # 梯度裁剪


# ===== 工具函数 =====

def ensure_model_dir() -> None:
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 GAE 优势和回报（returns）.

    参数：
        rewards: [T]
        values:  [T+1]，最后一个为 bootstrap value
        dones:   [T] (1.0 表示 episode 结束)
    """
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


# ===== 采样轨迹 =====

def collect_trajectories(
    env: TradingEnvV22MultiTF,
    policy: ActorCritic,
    batch_size: int,
) -> Tuple[np.ndarray, ...]:
    """
    与环境交互，采样一批轨迹数据，直到 time steps >= batch_size.

    返回：
        states      [T, state_dim]
        actions     [T]
        log_probs   [T]
        rewards     [T]
        dones       [T]
        values      [T+1]
    """
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

    # 末尾 bootstrap 一个 value
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


# ===== PPO 训练主循环 =====

def train_ppo(
    symbol: str,
    days: int,
    total_steps: int,
    config: PPOConfig,
    save_path: str,
) -> None:
    """
    PPO 训练入口。
    """
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
        # 1) 采样一批轨迹
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

        # 2) GAE + returns
        adv_np, returns_np = compute_gae(
            rewards=rewards_np,
            values=values_np,
            dones=dones_np,
            gamma=config.gamma,
            lam=config.lam,
        )
        adv_np = normalize(adv_np)

        # 转 Tensor
        states = torch.tensor(states_np, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions_np, dtype=torch.int64, device=DEVICE)
        old_log_probs = torch.tensor(log_probs_old_np, dtype=torch.float32, device=DEVICE)
        returns_t = torch.tensor(returns_np, dtype=torch.float32, device=DEVICE)
        adv_t = torch.tensor(adv_np, dtype=torch.float32, device=DEVICE)

        # 3) PPO 多轮更新
        policy.train()
        num_samples = len(states)
        idxs = np.arange(num_samples)

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

                loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                optimizer.step()

        elapsed = time.time() - start_time
        logger.info(
            f"[PPO] 已采样步数: {steps_collected}/{total_steps}, "
            f"本轮样本数={steps_this_iter}, 用时={elapsed:.1f}s"
        )

    # 训练结束，保存模型
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


# ===== 评估逻辑 =====

def evaluate_model(
    model_path: str,
    symbol: str,
    days: int,
    max_steps: int = 1000,
) -> None:
    """
    加载已训练模型，在环境中跑一条 episode。
    """
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
                f"[EVAL] step={step_count}, action={action}, "
                f"reward={reward:.6f}, equity={info['equity']:.2f}"
            )

    logger.info(
        f"[EVAL] 评估结束: 步数={step_count}, 总reward={total_reward:.4f}, "
        f"最终权益={info['equity']:.2f}"
    )


# ===== 命令行入口 =====

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"AI Trader PPO 训练器 ({VERSION_TAG})"
    )
    p.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="训练或评估的交易品种，例如 BTCUSDT",
    )
    p.add_argument(
        "--days",
        type=int,
        default=365,
        help="使用最近多少天的数据",
    )
    p.add_argument(
        "--total-steps",
        type=int,
        default=20000,
        help="训练时需要采样的总步数（越大越充分，但耗时也越长）",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="运行模式：train=训练, eval=评估已训练模型",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default="",
        help="评估模式下，指定要加载的模型路径；训练模式下可留空（自动生成）。",
    )
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
        train_ppo(
            symbol=symbol,
            days=days,
            total_steps=args.total_steps,
            config=config,
            save_path=save_path,
        )
        # 训练完自动评估一次
        evaluate_model(save_path, symbol=symbol, days=days, max_steps=1000)
    else:
        model_path = args.model_path or default_model_path
        evaluate_model(model_path, symbol=symbol, days=days, max_steps=1000)


if __name__ == "__main__":
    main()
