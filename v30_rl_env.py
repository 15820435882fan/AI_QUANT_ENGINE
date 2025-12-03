# v30_rl_env.py
#
# V30 强化学习环境 v2（混合型收益 + 风控奖励）
#
# 特点：
#   - 多周期特征：1h + 15m + 5m（由 v30_mtf_feature_builder 提供）
#   - 状态：最近 seq_len 根特征序列，shape = [seq_len, feature_dim]
#   - 动作：0 HOLD, 1 LONG, 2 SHORT, 3 CLOSE
#   - 账户模型：USDT 本位，单向持仓，多头 / 空头，固定风险比例开仓
#   - 奖励：混合型（C 方案）
#       reward = 0.7 * pnl_return   # 收益
#              + 0.2 * (-drawdown)  # 回撤惩罚
#              + 0.1 * trade_term   # 交易频率轻微惩罚
#
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np

from v30_mtf_feature_builder import build_mtf_features


@dataclass
class V30RLEnvConfig:
    symbol: str = "BTCUSDT"
    base_timeframe: str = "1h"
    mid_timeframe: str = "15m"
    fast_timeframe: str = "5m"
    days: int = 365
    seq_len: int = 128
    episode_steps: int = 512
    initial_equity: float = 10_000.0

    # 交易与风控参数
    fee_rate: float = 0.0004          # 双边手续费 0.04%
    slippage: float = 0.0002          # 预估滑点（占价格比例）
    risk_per_trade: float = 0.3       # 每次开仓使用账户资金的 30%

    # 奖励权重（C 方案：收益 + 风控 + 交易）
    w_pnl: float = 0.7
    w_dd: float = 0.2
    w_trades: float = 0.1
    trade_penalty: float = 0.001      # 每次换方向/平仓的轻微惩罚


class V30RLEnv:
    """
    强化学习环境：
      - 离线历史数据，适合 PPO 等 on-policy 算法
      - 每个 episode 从历史中随机截取一段，长度为 episode_steps
      - status = 最近 seq_len 个特征序列
    """

    def __init__(self, cfg: Optional[V30RLEnvConfig] = None):
        if cfg is None:
            cfg = V30RLEnvConfig()
        self.cfg = cfg

        df_mtf, feat_cols = build_mtf_features(
            symbol=cfg.symbol,
            base_timeframe=cfg.base_timeframe,
            mid_timeframe=cfg.mid_timeframe,
            fast_timeframe=cfg.fast_timeframe,
            days=cfg.days,
        )
        self.df = df_mtf
        self.feat_cols = feat_cols
        self.feats = df_mtf[feat_cols].values.astype(np.float32)
        # 使用基础周期收盘价作为结算价格
        self.prices = df_mtf["close"].values.astype(np.float32)
        self.timestamps = df_mtf.index.values
        self.n = len(self.df)

        if self.n <= self.cfg.seq_len + 2:
            raise ValueError(f"数据太短: n={self.n}, seq_len={self.cfg.seq_len}")

        # 交易状态 / 账户
        self.cash: float = cfg.initial_equity
        self.pos_side: int = 0          # 0 flat, 1 long, -1 short
        self.position_size: float = 0.0 # 当前持仓数量（BTC 数量）
        self.entry_price: float = 0.0

        self.equity: float = cfg.initial_equity
        self.peak_equity: float = cfg.initial_equity

        # 轨迹位置
        self.t: int = 0
        self.start_idx: int = 0

    # --------- 工具函数 ---------
    def _get_price(self, idx: int) -> float:
        idx = max(0, min(int(idx), self.n - 1))
        return float(self.prices[idx])

    def _mark_to_market(self, price: float) -> float:
        """
        根据当前价格计算总权益：现金 + 持仓市值（多头为正，空头为负）
        """
        pos_value = self.position_size * self.pos_side * price
        return self.cash + pos_value

    def _get_state(self) -> np.ndarray:
        start = self.t - self.cfg.seq_len
        if start < 0:
            pad_len = -start
            start = 0
        else:
            pad_len = 0

        end = self.t
        seq = self.feats[start:end]

        if pad_len > 0:
            pad = np.repeat(self.feats[[0]], pad_len, axis=0)
            seq = np.concatenate([pad, seq], axis=0)

        if len(seq) < self.cfg.seq_len:
            pad_len2 = self.cfg.seq_len - len(seq)
            pad2 = np.repeat(seq[[-1]], pad_len2, axis=0)
            seq = np.concatenate([seq, pad2], axis=0)

        return seq.astype(np.float32)

    def _reset_account(self):
        self.cash = self.cfg.initial_equity
        self.pos_side = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.equity = self.cfg.initial_equity
        self.peak_equity = self.cfg.initial_equity

    # --------- Gym-like API ---------
    def reset(self) -> np.ndarray:
        """
        随机选择一个 episode 起点，重置账户与时间位置，返回初始状态。
        """
        self._reset_account()
        # 为保证有 seq_len+episode_steps 的空间，起始点范围如下
        max_start = self.n - (self.cfg.seq_len + self.cfg.episode_steps + 1)
        if max_start < 0:
            max_start = 0
        self.start_idx = int(np.random.randint(0, max_start + 1))
        self.t = self.start_idx + self.cfg.seq_len

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行一步：
          action: 0 HOLD, 1 LONG, 2 SHORT, 3 CLOSE
        返回: next_state, reward, done, info
        """
        action = int(action)
        action = max(0, min(action, 3))

        # 当前价格（用于本步结算）
        price = self._get_price(self.t)

        # 当前权益（执行动作前）
        prev_equity = self._mark_to_market(price)

        # ---- 执行交易逻辑 ----
        desired_side = self.pos_side
        if action == 0:   # HOLD
            desired_side = self.pos_side
        elif action == 1: # LONG
            desired_side = 1
        elif action == 2: # SHORT
            desired_side = -1
        elif action == 3: # CLOSE
            desired_side = 0

        # 成交带滑点的价格
        trade_price = price * (1.0 + self.cfg.slippage * (1 if desired_side != self.pos_side else 0))

        # 1）如果需要平掉原有仓位
        if self.pos_side != 0 and desired_side != self.pos_side:
            # 平仓
            volume_close = abs(self.position_size) * trade_price
            # 多头：pos_side=1，空头：pos_side=-1
            pnl_close = self.position_size * self.pos_side * (trade_price - self.entry_price)
            fee_close = volume_close * self.cfg.fee_rate

            self.cash += pnl_close - fee_close
            self.position_size = 0.0
            self.pos_side = 0
            self.entry_price = 0.0

        # 2）如果需要开新仓（多 / 空）
        if desired_side != 0 and self.pos_side == 0:
            # 用当前现金的 risk_per_trade 比例建立仓位
            trade_cash = self.cash * self.cfg.risk_per_trade
            if trade_cash > 0:
                new_size = trade_cash / trade_price
                volume_open = trade_cash
                fee_open = volume_open * self.cfg.fee_rate

                self.cash -= fee_open
                self.position_size = new_size
                self.pos_side = desired_side
                self.entry_price = trade_price

        # 若只是保持原有仓位且 action=0，什么都不做（被动持有）

        # 3）用本步价格重新 mark-to-market
        self.equity = self._mark_to_market(price)
        self.peak_equity = max(self.peak_equity, self.equity)

        # ---- 计算奖励（混合型）----
        raw_pnl = self.equity - prev_equity
        # 用当前权益尺度归一化收益（避免数值过大过小）
        ref_equity = max(prev_equity, 1e-6)
        pnl_return = raw_pnl / ref_equity

        # 回撤：相对于历史峰值的回撤比例
        if self.peak_equity > 1e-6:
            drawdown = max(0.0, (self.peak_equity - self.equity) / self.peak_equity)
        else:
            drawdown = 0.0

        # 交易频率惩罚：每次发生「换方向 / 平仓 / 新开仓」给予轻微惩罚
        trade_term = 0.0
        if action in (1, 2, 3):
            trade_term = -self.cfg.trade_penalty

        reward = (
            self.cfg.w_pnl * pnl_return +
            self.cfg.w_dd * (-drawdown) +
            self.cfg.w_trades * trade_term
        )

        # ---- 时间推进 & 终止条件 ----
        self.t += 1
        done = False
        if self.t >= self.start_idx + self.cfg.episode_steps or self.t >= self.n - 1:
            done = True

        next_state = self._get_state()
        info = {
            "equity": float(self.equity),
            "pos_side": int(self.pos_side),
            "position_size": float(self.position_size),
            "t_index": int(self.t),
            "timestamp": self.timestamps[self.t] if self.t < len(self.timestamps) else None,
            "price": float(price),
            "pnl_return": float(pnl_return),
            "drawdown": float(drawdown),
        }
        return next_state, float(reward), done, info
