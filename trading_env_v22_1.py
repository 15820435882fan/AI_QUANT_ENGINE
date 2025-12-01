"""
trading_env_v22_1.py

V22_1: AI Trader 强化学习环境 · 第一版（Gym 风格）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

INITIAL_CAPITAL = 10_000.0
TRADING_FEE_RATE = 0.0005  # 单边手续费 0.05%
SLIPPAGE_RATE = 0.0002     # 简单滑点 0.02%

try:
    from local_data_engine import load_local_kline
except Exception:
    def load_local_kline(*args, **kwargs):
        raise RuntimeError("未找到 local_data_engine.load_local_kline，请确认同目录下存在该文件。")


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


@dataclass
class Position:
    side: int  # -1: short, 0: flat, 1: long
    leverage: float
    entry_price: float
    size: float  # 名义头寸 = equity * leverage / price


class TradingEnvV22:
    """
    强化学习交易环境（单标的、1h）。
    """

    def __init__(
        self,
        symbol: str,
        days: int = 365,
        start_index: int = 200,
        max_steps: Optional[int] = None,
    ):
        self.symbol = symbol
        self.days = days
        self.start_index = start_index
        self.max_steps = max_steps

        self.df: Optional[pd.DataFrame] = None
        self.current_step: int = 0
        self.equity: float = INITIAL_CAPITAL
        self.initial_price: float = 0.0
        self.position: Position = Position(side=0, leverage=0.0, entry_price=0.0, size=0.0)
        self.done: bool = False

        self.prev_equity: float = INITIAL_CAPITAL
        self.steps_in_episode: int = 0

        self._load_data()
        self._prepare_features()

        self.action_space_n = 4  # 0:hold, 1:long, 2:short, 3:flat

    def _load_data(self) -> None:
        logger.info(f"[EnvV22_1] 加载数据: {self.symbol}, 1h, days={self.days}")
        raw = load_local_kline(self.symbol, "1h", self.days)
        df = raw.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="ignore")
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df = df.set_index("timestamp")
        df = df.sort_index()

        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                raise ValueError(f"{self.symbol} 1h 数据缺少列: {col}")

        self.df = df[["open", "high", "low", "close"]].copy()

    def _prepare_features(self) -> None:
        df = self.df.copy()
        close = df["close"]

        df["ret_1"] = close.pct_change(1)
        df["ret_6"] = close.pct_change(6)
        df["ret_24"] = close.pct_change(24)

        ema_fast = close.ewm(span=20, adjust=False).mean()
        ema_slow = close.ewm(span=60, adjust=False).mean()
        df["ema_fast_ratio"] = ema_fast / close - 1.0
        df["ema_slow_ratio"] = ema_slow / close - 1.0

        high = df["high"]
        low = df["low"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()
        df["atr_ratio"] = df["atr_14"] / close

        df["close_norm"] = close / close.iloc[0]

        self.df = df.dropna().copy()

        if self.start_index < 5:
            self.start_index = 5
        if self.start_index >= len(self.df) - 10:
            self.start_index = max(5, len(self.df) // 3)

    def reset(self) -> np.ndarray:
        self.done = False
        self.equity = INITIAL_CAPITAL
        self.prev_equity = INITIAL_CAPITAL
        self.position = Position(side=0, leverage=0.0, entry_price=0.0, size=0.0)
        self.steps_in_episode = 0

        self.current_step = self.start_index
        self.initial_price = float(self.df["close"].iloc[self.current_step])

        obs = self._get_state()
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            raise RuntimeError("请先调用 reset() 再继续 step。")

        if action not in (0, 1, 2, 3):
            raise ValueError(f"非法动作: {action}")

        price = float(self.df["close"].iloc[self.current_step])

        self._apply_action(action, price)

        self.prev_equity = self.equity
        self.current_step += 1
        self.steps_in_episode += 1

        next_price = float(self.df["close"].iloc[self.current_step])
        self._mark_to_market(next_price)

        reward = (self.equity - self.prev_equity) / INITIAL_CAPITAL

        if self.current_step >= len(self.df) - 2:
            self.done = True
        if self.max_steps is not None and self.steps_in_episode >= self.max_steps:
            self.done = True
        if self.equity <= INITIAL_CAPITAL * 0.1:
            self.done = True

        obs = self._get_state()
        info = {
            "equity": self.equity,
            "prev_equity": self.prev_equity,
            "price": next_price,
            "step": self.steps_in_episode,
            "position_side": self.position.side,
            "position_leverage": self.position.leverage,
            "position_size": self.position.size,
        }
        return obs, float(reward), self.done, info

    def action_space_sample(self) -> int:
        return int(np.random.randint(0, self.action_space_n))

    def _apply_action(self, action: int, price: float) -> None:
        pos = self.position

        if action == 0:
            return

        if action == 3 and pos.side != 0:
            self._close_position(price)
            return

        if action == 1 and pos.side == 0:
            self._open_position(side=1, price=price)
            return

        if action == 2 and pos.side == 0:
            self._open_position(side=-1, price=price)
            return

        return

    def _open_position(self, side: int, price: float) -> None:
        target_leverage = 2.0
        notional = self.equity * target_leverage
        size = notional / price

        fee = notional * TRADING_FEE_RATE
        slippage_price = price * (1 + SLIPPAGE_RATE * side)

        if fee >= self.equity * 0.1:
            return

        self.equity -= fee

        self.position = Position(
            side=side,
            leverage=target_leverage,
            entry_price=slippage_price,
            size=size,
        )

    def _close_position(self, price: float) -> None:
        pos = self.position
        if pos.side == 0 or pos.size <= 0:
            return

        slippage_price = price * (1 - SLIPPAGE_RATE * pos.side)

        notional_entry = pos.entry_price * pos.size
        notional_exit = slippage_price * pos.size

        if pos.side > 0:
            pnl = notional_exit - notional_entry
        else:
            pnl = notional_entry - notional_exit

        fee = notional_exit * TRADING_FEE_RATE

        self.equity += pnl
        self.equity -= fee

        self.position = Position(side=0, leverage=0.0, entry_price=0.0, size=0.0)

    def _mark_to_market(self, price: float) -> None:
        pos = self.position
        if pos.side == 0 or pos.size <= 0:
            return
        # 当前版本中，为简化处理，浮盈不额外修改 equity，
        # 只在平仓时结算实际 pnl。后续版本中可改为完整 MTM。
        return

    def _get_state(self) -> np.ndarray:
        idx = self.current_step
        row = self.df.iloc[idx]

        close_norm = float(row["close_norm"])
        ret_1 = float(row["ret_1"])
        ret_6 = float(row["ret_6"])
        ret_24 = float(row["ret_24"])
        ema_fast_ratio = float(row["ema_fast_ratio"])
        ema_slow_ratio = float(row["ema_slow_ratio"])
        atr_ratio = float(row["atr_ratio"])

        pos = self.position
        position_side = float(pos.side)
        position_leverage_norm = float(pos.leverage / 5.0)
        if pos.size > 0 and self.equity > 0:
            position_value = pos.size * float(row["close"])
            position_value_ratio = float(position_value / self.equity)
        else:
            position_value_ratio = 0.0

        unrealized_pnl_ratio = 0.0
        equity_ratio = float(self.equity / INITIAL_CAPITAL)

        state = np.array(
            [
                close_norm,
                ret_1,
                ret_6,
                ret_24,
                ema_fast_ratio,
                ema_slow_ratio,
                atr_ratio,
                position_side,
                position_leverage_norm,
                position_value_ratio,
                unrealized_pnl_ratio,
                equity_ratio,
            ],
            dtype=np.float32,
        )
        return state


if __name__ == "__main__":
    env = TradingEnvV22(symbol="BTCUSDT", days=365)
    obs = env.reset()
    logger.info(f"初始 obs 维度: {obs.shape}, 值示例: {obs}")

    done = False
    step_count = 0
    total_reward = 0.0

    while not done and step_count < 200:
        action = env.action_space_sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1

    logger.info(
        f"随机策略跑完 {step_count} 步, 总 reward={total_reward:.4f}, "
        f"最终权益={info.get('equity', None):.2f}"
    )
