# v30_ppo_finetune_v3.py
#
# V30 · Step5 专业版 PPO 微调脚本（带训练进度、日志、曲线）
#
# 功能升级：
#   1）自动迁移监督版 Transformer（跳过 input_proj，解决 9→MTF 输入维度不匹配）
#   2）接入多周期 RL 环境：1h + 15m + 5m（来自 v30_rl_env / v30_mtf_feature_builder）
#   3）训练过程记录到 CSV：models_v30/ppo_train_log.csv
#   4）训练完成自动生成奖励曲线：models_v30/ppo_reward_curve.png
#   5）带简易进度条 + ETA 时间估算 + 动作分布统计
#
from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import matplotlib.pyplot as plt

from v30_model import build_v30_transformer
from v30_rl_env import V30RLEnv, V30RLEnvConfig


# ---------------- PPO 配置 ----------------
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
    train_transformer: bool = True  # 是否微调 transformer 主体


# ---------------- Actor-Critic 模型 ----------------
class V30ActorCritic(nn.Module):
    """
    基于 v30 Transformer 的 Actor-Critic：
    - Transformer：抽取时序特征
    - policy_head：输出动作 logits
    - value_head：输出状态价值
    """

    def __init__(self, seq_len: int, feature_dim: int, num_actions: int = 4):
        super().__init__()
        self.transformer = build_v30_transformer(
            feature_dim=feature_dim,
            num_actions=num_actions,
            num_risks=3,
            seq_len=seq_len,
        )
        self.policy_head = self.transformer.action_head
        d_model = self.transformer.config.d_model
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, F]
        返回:
            logits: [B, num_actions]
            values: [B, 1]
        """
        h = self.transformer.input_proj(x)
        h = self.transformer.pos_encoder(h)
        h = self.transformer.transformer(h)
        h = self.transformer.norm(h)
        ctx = h[:, -1, :]
        logits = self.policy_head(ctx)
        values = self.value_head(ctx)
        return logits, values


# ---------------- 加载监督版 Transformer（迁移学习） ----------------
def load_pretrained_actor_critic(
    seq_len: int,
    feature_dim: int,
    num_actions: int,
    ckpt_path: str,
    device: torch.device,
    train_transformer: bool = True,
) -> V30ActorCritic:
    """
    从 v30_train_supervised 的 best.pt 加载 Transformer 权重。
    关键修复：
      - 自动过滤掉 input_proj.* 权重（9 维 → MTF 维度不一致）
    """
    model = V30ActorCritic(seq_len=seq_len, feature_dim=feature_dim, num_actions=num_actions)

    print(f"[PPO Loader] 加载监督模型: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"]

    remove_keys = [k for k in state_dict.keys() if k.startswith("input_proj.")]
    if remove_keys:
        print(f"[PPO Loader] ⚠ 移除不兼容 input_proj 权重 {len(remove_keys)} 项")
        for k in remove_keys:
            del state_dict[k]

    model.transformer.load_state_dict(state_dict, strict=False)
    model.to(device)

    if not train_transformer:
        for p in model.transformer.parameters():
            p.requires_grad = False

    print("[PPO Loader] ✓ 监督权重迁移完成（input_proj 将在 MTF 环境中重新学习）")
    return model


# ---------------- GAE-Lambda ----------------
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GAE-Lambda 计算优势与回报。
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


# ---------------- PPO 更新 ----------------
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

    device = obs.device
    n = obs.size(0)
    idxs = torch.randperm(n, device=device)

    policy_losses, value_losses, entropies = [], [], []

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

    return (
        torch.stack(policy_losses).mean().item(),
        torch.stack(value_losses).mean().item(),
        torch.stack(entropies).mean().item(),
    )


# ---------------- 辅助：进度条与时间格式 ----------------
def format_td(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ---------------- PPO 训练主流程 ----------------
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

    if cfg is None:
        cfg = PPOConfig()

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if cfg.device == "auto" else torch.device(cfg.device)
    print(f"[V30 PPO] 使用 device={device}")

    # 环境（多周期）
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
    print(f"[V30 PPO] 特征维度 feature_dim={feature_dim}")

    num_actions = 4

    # 模型与优化器
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

    # 日志路径
    models_dir = "models_v30"
    ensure_dir(models_dir)
    csv_path = os.path.join(models_dir, "ppo_train_log.csv")
    png_path = os.path.join(models_dir, "ppo_reward_curve.png")

    # CSV 初始化
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step",
                "episode",
                "reward_mean",
                "policy_loss",
                "value_loss",
                "entropy",
                "equity_mean",
                "hold_ratio",
                "long_ratio",
                "short_ratio",
                "close_ratio",
            ])

    obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, eq_buf = [], [], [], [], [], [], []

    total_steps = 0
    episode_count = 0
    start_time = time.time()

    # 用于画 reward 曲线
    reward_curve_steps = []
    reward_curve_values = []

    obs = env.reset()

    while total_steps < cfg.total_steps:
        # ===== 收集一段 rollouts =====
        action_hist = {0: 0, 1: 0, 2: 0, 3: 0}

        for _ in range(cfg.rollout_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

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
            eq_buf.append(float(info.get("equity", 0.0)))

            action_hist[action_int] = action_hist.get(action_int, 0) + 1

            obs = next_obs
            total_steps += 1

            if done:
                episode_count += 1
                obs = env.reset()

            if total_steps >= cfg.total_steps:
                break

        # ===== 转为 tensor =====
        obs_tensor = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(np.array(act_buf), dtype=torch.int64, device=device)
        logprobs_old_tensor = torch.tensor(np.array(logp_buf), dtype=torch.float32, device=device)
        rewards_tensor = torch.tensor(np.array(rew_buf), dtype=torch.float32, device=device)
        dones_tensor = torch.tensor(np.array(done_buf), dtype=torch.float32, device=device)
        values_tensor = torch.tensor(np.array(val_buf), dtype=torch.float32, device=device)
        equity_tensor = torch.tensor(np.array(eq_buf), dtype=torch.float32, device=device)

        # 最后一个状态的 value
        obs_last_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, last_value = actor_critic(obs_last_t)
        last_val_scalar = float(last_value.squeeze(-1).item())
        last_val_tensor = torch.tensor(last_val_scalar, device=device, dtype=torch.float32)

        # GAE
        advantages, returns = compute_gae(
            rewards=rewards_tensor,
            values=values_tensor,
            dones=dones_tensor,
            last_value=last_val_tensor,
            gamma=cfg.gamma,
            lam=cfg.lam,
        )

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO 更新
        pl, vl, ent = ppo_update(
            actor_critic,
            optimizer,
            obs_tensor,
            actions_tensor,
            logprobs_old_tensor,
            returns,
            advantages,
            cfg,
        )

        avg_reward = rewards_tensor.mean().item()
        avg_equity = equity_tensor.mean().item() if equity_tensor.numel() > 0 else 0.0

        # 行为统计
        total_actions = sum(action_hist.values()) or 1
        hold_ratio = action_hist.get(0, 0) / total_actions
        long_ratio = action_hist.get(1, 0) / total_actions
        short_ratio = action_hist.get(2, 0) / total_actions
        close_ratio = action_hist.get(3, 0) / total_actions

        # 写入 CSV
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                total_steps,
                episode_count,
                avg_reward,
                pl,
                vl,
                ent,
                avg_equity,
                hold_ratio,
                long_ratio,
                short_ratio,
                close_ratio,
            ])

        # 更新进度条 & ETA
        elapsed = time.time() - start_time
        steps_per_sec = total_steps / max(elapsed, 1e-6)
        remaining_steps = max(cfg.total_steps - total_steps, 0)
        eta = remaining_steps / max(steps_per_sec, 1e-6)

        progress = total_steps / cfg.total_steps
        bar_len = 20
        filled = int(progress * bar_len)
        bar = "█" * filled + "-" * (bar_len - filled)

        print(
            f"[V30 PPO] |{bar}| {progress*100:5.1f}% "
            f"steps={total_steps}/{cfg.total_steps} "
            f"ETA={format_td(eta)} "
            f"avg_reward={avg_reward:.6f} "
            f"ent={ent:.4f} "
            f"act(H/L/S/C)={hold_ratio:.2f}/{long_ratio:.2f}/{short_ratio:.2f}/{close_ratio:.2f}"
        )

        # 记录 reward 曲线
        reward_curve_steps.append(total_steps)
        reward_curve_values.append(avg_reward)

        # 清空缓冲区
        obs_buf.clear()
        act_buf.clear()
        logp_buf.clear()
        rew_buf.clear()
        done_buf.clear()
        val_buf.clear()
        eq_buf.clear()

    # ===== 训练结束：保存模型 & 画图 =====
    ckpt = {
        "symbol": symbol,
        "timeframe": timeframe,
        "seq_len": seq_len,
        "feature_dim": feature_dim,
        "model_state_dict": actor_critic.transformer.state_dict(),
    }
    torch.save(ckpt, out_path)
    print(f"[V30 PPO] 微调完成，已保存到: {out_path}")

    # 绘制奖励曲线（深灰底 + 亮青色线风格）
    if reward_curve_steps:
        plt.figure(figsize=(10, 5))
        plt.style.use("dark_background")
        plt.plot(reward_curve_steps, reward_curve_values, linewidth=1.5)
        plt.xlabel("Training Steps")
        plt.ylabel("Avg Reward per Rollout")
        plt.title("V30 PPO Training Reward Curve (MTF 1h+15m+5m)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        print(f"[V30 PPO] 奖励曲线已保存到: {png_path}")
        # 自动弹出图像
        plt.show()

    return out_path


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="V30 PPO Finetune v3 (with logs & curves)")
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--timeframe", type=str, default="1h")
    p.add_argument("--pretrained-path", type=str, default="")
    p.add_argument("--out-path", type=str, default="")
    p.add_argument("--total-steps", type=int, default=200000)
    p.add_argument("--rollout-steps", type=int, default=1024)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--train-transformer", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    symbol = args.symbol.upper()
    timeframe = args.timeframe

    pretrained_path = args.pretrained_path or f"models_v30/v30_transformer_{symbol}_{timeframe}_best.pt"
    out_path = args.out_path or f"models_v30/v30_transformer_{symbol}_{timeframe}_ppo_v3.pt"

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
