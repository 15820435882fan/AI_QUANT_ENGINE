# v30_ppo_finetune.py
#
# 使用 PPO 对 V30 Transformer 模型进行收益导向的微调（强化学习）。
#
# 思路：
#   - 使用 V30RLEnv 环境（多周期特征：1h + 15m + 5m）
#   - 使用已经监督训练好的 Transformer 作为特征提取 backbone
#   - 新增一个 value head，用于 PPO 的 critic
#   - 可选择是否微调整个 Transformer 或只训练最后几层
#
# 示例用法（在项目根目录运行）：
#
#   python v30_ppo_finetune.py --symbol BTCUSDT --timeframe 1h \
#       --pretrained-path models_v30/v30_transformer_BTCUSDT_1h_best.pt \
#       --total-steps 200000
#
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from v30_model import build_v30_transformer
from v30_rl_env import V30RLEnv, V30RLEnvConfig


@dataclass
class PPOConfig:
    total_steps: int = 200_000
    rollout_steps: int = 1024
    batch_size: int = 256
    gamma: float = 0.99
    lam: float = 0.95
    ppo_epochs: int = 5
    clip_ratio: float = 0.2
    lr: float = 3e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    device: str = "auto"
    train_transformer: bool = True  # 是否微调 Transformer 主体


class V30ActorCritic(nn.Module):
    """
    基于 V30 Transformer 的 Actor-Critic：
      - Transformer 提取时序特征
      - policy_head 输出动作 logits
      - value_head 输出状态价值
    """

    def __init__(self, seq_len: int, feature_dim: int, num_actions: int = 4):
        super().__init__()
        self.transformer = build_v30_transformer(
            feature_dim=feature_dim,
            num_actions=num_actions,
            num_risks=3,
            seq_len=seq_len,
        )
        # policy head 直接复用 transformer 的 action_head
        self.policy_head = self.transformer.action_head
        # 新增 value head
        d_model = self.transformer.config.d_model
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, F]
        返回：
          logits: [B, num_actions]
          values: [B, 1]
        """
        h = self.transformer.input_proj(x)
        h = self.transformer.pos_encoder(h)
        h = self.transformer.transformer(h)
        h = self.transformer.norm(h)
        ctx = h[:, -1, :]  # 取最后一个时间步作为总体表示
        logits = self.policy_head(ctx)
        values = self.value_head(ctx)
        return logits, values


def load_pretrained_actor_critic(
    seq_len: int,
    feature_dim: int,
    num_actions: int,
    ckpt_path: str,
    device: torch.device,
    train_transformer: bool = True,
) -> V30ActorCritic:
    """
    从监督学习的 checkpoint 中加载 Transformer 权重，初始化 Actor-Critic。
    """
    model = V30ActorCritic(seq_len=seq_len, feature_dim=feature_dim, num_actions=num_actions)
    ckpt = torch.load(ckpt_path, map_location=device)
    # strict=False 的原因：新加了 value_head，不在原有 state_dict 中
    model.transformer.load_state_dict(ckpt["model_state_dict"], strict=False)

    model.to(device)

    if not train_transformer:
        for p in model.transformer.parameters():
            p.requires_grad = False

    return model


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GAE-Lambda 计算优势和回报。
      rewards: [T]
      values:  [T]
      dones:   [T]
      last_value: 标量（最后一个状态的 value）
    返回：
      advantages: [T]
      returns:    [T]
    """
    T = rewards.size(0)
    advantages = torch.zeros(T, dtype=torch.float32, device=rewards.device)
    last_adv = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


def ppo_update(
    actor_critic: V30ActorCritic,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    logprobs_old: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    cfg: PPOConfig,
) -> Tuple[float, float, float]:
    """
    对一个大 batch 的经验进行多 epoch PPO 更新。
    返回：
      平均 policy_loss, value_loss, entropy
    """
    device = obs.device
    n = obs.size(0)
    idxs = torch.randperm(n, device=device)

    policy_losses = []
    value_losses = []
    entropies = []

    for _ in range(cfg.ppo_epochs):
        for start in range(0, n, cfg.batch_size):
            end = start + cfg.batch_size
            batch_idx = idxs[start:end]

            b_obs = obs[batch_idx]
            b_actions = actions[batch_idx]
            b_logprobs_old = logprobs_old[batch_idx]
            b_returns = returns[batch_idx]
            b_advantages = advantages[batch_idx]

            logits, values = actor_critic(b_obs)
            dist = Categorical(logits=logits)
            logprobs = dist.log_prob(b_actions)
            entropy = dist.entropy().mean()

            # PPO ratio
            ratio = (logprobs - b_logprobs_old).exp()
            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * b_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (b_returns - values.squeeze(-1)).pow(2).mean()

            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor_critic.parameters(), cfg.max_grad_norm)
            optimizer.step()

            policy_losses.append(policy_loss.detach())
            value_losses.append(value_loss.detach())
            entropies.append(entropy.detach())

    pl = torch.stack(policy_losses).mean().item()
    vl = torch.stack(value_losses).mean().item()
    ent = torch.stack(entropies).mean().item()
    return pl, vl, ent


def train_ppo(
    symbol: str,
    timeframe: str,
    pretrained_path: str,
    out_path: str,
    env_days: int = 365,
    env_episode_steps: int = 512,
    seq_len: int = 128,
    cfg: PPOConfig | None = None,
) -> str:
    """
    在 V30RLEnv 上使用 PPO 对预训练 Transformer 进行微调。
    返回：
      微调后 checkpoint 路径
    """
    if cfg is None:
        cfg = PPOConfig()

    # 设备
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    print(f"[V30 PPO] device={device}")

    # 环境
    env_cfg = V30RLEnvConfig(
        symbol=symbol,
        base_timeframe=timeframe,
        mid_timeframe="15m",
        fast_timeframe="5m",
        days=env_days,
        seq_len=seq_len,
        episode_steps=env_episode_steps,
        initial_equity=10_000.0,
    )
    env = V30RLEnv(env_cfg)

    feature_dim = env.feats.shape[1]
    num_actions = 4

    # 模型
    actor_critic = load_pretrained_actor_critic(
        seq_len=seq_len,
        feature_dim=feature_dim,
        num_actions=num_actions,
        ckpt_path=pretrained_path,
        device=device,
        train_transformer=cfg.train_transformer,
    )

    optimizer = torch.optim.Adam(
        [p for p in actor_critic.parameters() if p.requires_grad],
        lr=cfg.lr,
    )

    obs_buf = []
    act_buf = []
    logp_buf = []
    rew_buf = []
    done_buf = []
    val_buf = []

    total_steps = 0
    episode_count = 0

    obs = env.reset()
    while total_steps < cfg.total_steps:
        # 收集一段 rollouts
        for _ in range(cfg.rollout_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T, F]
            with torch.no_grad():
                logits, value = actor_critic(obs_t)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)

            action_int = int(action.item())
            next_obs, reward, done, info = env.step(action_int)

            obs_buf.append(obs)
            act_buf.append(action_int)
            logp_buf.append(float(logprob.item()))
            rew_buf.append(float(reward))
            done_buf.append(float(done))
            val_buf.append(float(value.item()))

            obs = next_obs
            total_steps += 1

            if done:
                episode_count += 1
                obs = env.reset()

            if total_steps >= cfg.total_steps:
                break

        # 将缓冲区转换为张量
        obs_tensor = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(np.array(act_buf), dtype=torch.int64, device=device)
        logprobs_old_tensor = torch.tensor(np.array(logp_buf), dtype=torch.float32, device=device)
        rewards_tensor = torch.tensor(np.array(rew_buf), dtype=torch.float32, device=device)
        dones_tensor = torch.tensor(np.array(done_buf), dtype=torch.float32, device=device)
        values_tensor = torch.tensor(np.array(val_buf), dtype=torch.float32, device=device)

        # 计算最后一个状态的 value（用于 GAE）
        obs_last_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, last_value = actor_critic(obs_last_t)
        last_value_scalar = last_value.squeeze(-1).item()
        last_value_tensor = torch.tensor(last_value_scalar, dtype=torch.float32, device=device)

        advantages, returns = compute_gae(
            rewards=rewards_tensor,
            values=values_tensor,
            dones=dones_tensor,
            last_value=last_value_tensor,
            gamma=cfg.gamma,
            lam=cfg.lam,
        )

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO 更新
        pl, vl, ent = ppo_update(
            actor_critic=actor_critic,
            optimizer=optimizer,
            obs=obs_tensor,
            actions=actions_tensor,
            logprobs_old=logprobs_old_tensor,
            returns=returns,
            advantages=advantages,
            cfg=cfg,
        )

        avg_reward = rewards_tensor.mean().item()
        print(
            f"[V30 PPO] steps={total_steps} episodes={episode_count} "
            f"avg_reward={avg_reward:.6f} policy_loss={pl:.4f} value_loss={vl:.4f} entropy={ent:.4f}"
        )

        # 清空缓冲区
        obs_buf.clear()
        act_buf.clear()
        logp_buf.clear()
        rew_buf.clear()
        done_buf.clear()
        val_buf.clear()

    # 只保存 Transformer 主体权重（便于后续在 v30_eval_in_env 中复用）
    ckpt = {
        "symbol": symbol,
        "timeframe": timeframe,
        "seq_len": seq_len,
        "feature_dim": feature_dim,
        "model_state_dict": actor_critic.transformer.state_dict(),
    }
    torch.save(ckpt, out_path)
    print(f"[V30 PPO] 微调完成，已保存到: {out_path}")
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description="V30 Transformer PPO 微调")
    p.add_argument("--symbol", type=str, default="BTCUSDT", help="交易对，例如 BTCUSDT")
    p.add_argument("--timeframe", type=str, default="1h", help="基础周期，例如 1h")
    p.add_argument(
        "--pretrained-path",
        type=str,
        default="",
        help="已监督训练好的 Transformer 模型路径（v30_train_supervised 的 best.pt）",
    )
    p.add_argument(
        "--out-path",
        type=str,
        default="",
        help="微调后模型保存路径（默认自动生成到 models_v30 下）",
    )
    p.add_argument("--total-steps", type=int, default=200000, help="PPO 训练总步数")
    p.add_argument("--rollout-steps", type=int, default=1024, help="每次 rollouts 收集的步数")
    p.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda")
    p.add_argument("--train-transformer", action="store_true", help="是否微调 Transformer 主体（默认只训练头部）")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    symbol = args.symbol.upper()
    timeframe = args.timeframe

    pretrained_path = args.pretrained_path
    if not pretrained_path:
        pretrained_path = f"models_v30/v30_transformer_{symbol}_{timeframe}_best.pt"

    out_path = args.out_path
    if not out_path:
        out_path = f"models_v30/v30_transformer_{symbol}_{timeframe}_ppo_finetuned.pt"

    cfg = PPOConfig(
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        device=args.device,
        train_transformer=args.train_transformer,
    )

    train_ppo(
        symbol=symbol,
        timeframe=timeframe,
        pretrained_path=pretrained_path,
        out_path=out_path,
        env_days=365,
        env_episode_steps=512,
        seq_len=128,
        cfg=cfg,
    )
