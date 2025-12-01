"""
trading_env_v22_4.py

V22_4: AI Trader 强化学习环境 · 多周期状态版 + 奖励强化版（Reward Shaping 2.0）

在 V22_2 的基础上做了两个关键升级：
1. ✅ 引入完整逐小时 MTM（Mark-to-Market）机制：
   - 账户拆分为 cash（现金）+ 持仓的未实现盈亏；
   - 每一个时间步都会根据价格变化动态更新 equity；
   - 即使不平仓，reward 也会随着行情变化而变化，AI 不再是“全 0 奖励”。

2. ✅ 重写奖励函数（Reward Shaping 2.0）：
   - 奖励项：当前步的 ΔEquity / INITIAL_CAPITAL（与实际赚钱直接挂钩）；
   - 惩罚项：轻微的时间惩罚 + 持仓敞口惩罚（鼓励有把握时才开仓、不乱重仓）；
   - 奖励被适度裁剪（clip），防止极端值干扰训练稳定性。

接口保持一致：
    class TradingEnvV22MultiTF:
        reset() -> obs
        step(action) -> (obs, reward, done, info)

动作定义：
    0: HOLD 不动
    1: 开多（空仓时，2x 杠杆）
    2: 开空（空仓时，2x 杠杆）
    3: 平仓（有仓位则全部平掉）

注意：
    - 该环境文件命名为 trading_env_v22_4.py，
      你需要在 PPO 训练脚本里将：
          from trading_env_v22_2 import TradingEnvV22MultiTF
      替换为：
          from trading_env_v22_4 import TradingEnvV22MultiTF
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
    size: float  # 名义头寸 = price * size，即 size 为“张数”/“数量”


def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """确保 DF 有 open/high/low/close，并按时间索引排序。"""
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="ignore")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.set_index("timestamp")
    df = df.sort_index()

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    return df[["open", "high", "low", "close"]].copy()


def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _zscore(series: pd.Series, window: int = 200) -> pd.Series:
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    return (series - m) / (s + 1e-9)


class TradingEnvV22MultiTF:
    """
    强化学习交易环境（单标的，多周期状态，1h 为主时间轴，Reward Shaping 2.0）。

    接口：
        reset() -> obs
        step(action) -> (obs, reward, done, info)
        action_space_n: 动作空间维度（当前为 4）
        action_space_sample(): 返回一个随机动作，用于测试环境

    动作空间（当前版本）：
        0: HOLD 不动
        1: 开多（空仓时，以固定杠杆开多）
        2: 开空（空仓时，以固定杠杆开空）
        3: 平仓（有仓位则全部平掉）
    """

    def __init__(
        self,
        symbol: str,
        days: int = 365,
        start_index: int = 300,
        max_steps: Optional[int] = None,
    ):
        """
        :param symbol: 交易品种，例如 "BTCUSDT"
        :param days: 使用最近多少天的数据（1h / 15m / 4h）
        :param start_index: 从第几根 1h K 线之后才允许开始（用于预热多周期特征）
        :param max_steps: episode 最大步数（默认直到数据结束）
        """
        self.symbol = symbol
        self.days = days
        self.start_index = start_index
        self.max_steps = max_steps

        # 原始多周期数据
        self.df_1h: Optional[pd.DataFrame] = None
        self.df_15m: Optional[pd.DataFrame] = None
        self.df_4h: Optional[pd.DataFrame] = None

        # 对齐到 1h 的多周期特征表
        self.df_state: Optional[pd.DataFrame] = None

        # 账户 & 仓位
        self.current_step: int = 0  # 指向 df_state 行号
        self.cash: float = INITIAL_CAPITAL  # 现金
        self.equity: float = INITIAL_CAPITAL  # 总权益 = 现金 + 未实现盈亏
        self.initial_price: float = 0.0
        self.position: Position = Position(side=0, leverage=0.0, entry_price=0.0, size=0.0)
        self.done: bool = False

        self.prev_equity: float = INITIAL_CAPITAL
        self.steps_in_episode: int = 0

        # 加载数据并构造状态特征
        self._load_data_multitf()
        self._build_multitf_state()

        self.action_space_n = 4  # 0:hold, 1:long, 2:short, 3:flat

        logger.info(f"[EnvV22_4] 状态维度: {self.state_dim}")

    # ========= 多周期数据 & 状态特征 =========

    def _load_data_multitf(self) -> None:
        logger.info(f"[EnvV22_4] 加载多周期数据: {self.symbol}, days={self.days}")

        raw_1h = load_local_kline(self.symbol, "1h", self.days)
        raw_15m = load_local_kline(self.symbol, "15m", self.days)
        raw_4h = load_local_kline(self.symbol, "4h", self.days + 60)  # 4h 稍微多取一点

        self.df_1h = _ensure_ohlc(raw_1h)
        self.df_15m = _ensure_ohlc(raw_15m)
        self.df_4h = _ensure_ohlc(raw_4h)

    def _build_multitf_state(self) -> None:
        """
        构建以 1h 为主时间轴的多周期特征表 df_state。
        包含：
            - 1h: 价格、收益、EMA、ATR、波动
            - 4h: 趋势强度 & 方向
            - 15m: 短周期动量与波动 + 突破
            - Regime: 趋势/震荡 状态
        """
        df1 = self.df_1h.copy()
        close_1h = df1["close"]

        # ---------- 1h 特征 ----------
        df1["ret_1"] = close_1h.pct_change(1)
        df1["ret_6"] = close_1h.pct_change(6)
        df1["ret_24"] = close_1h.pct_change(24)

        ema_fast_1h = close_1h.ewm(span=20, adjust=False).mean()
        ema_slow_1h = close_1h.ewm(span=60, adjust=False).mean()
        df1["ema_fast_ratio"] = ema_fast_1h / close_1h - 1.0
        df1["ema_slow_ratio"] = ema_slow_1h / close_1h - 1.0

        atr_14_1h = _calc_atr(df1, period=14)
        df1["atr_14"] = atr_14_1h
        df1["atr_ratio"] = atr_14_1h / close_1h

        df1["vol_24"] = close_1h.pct_change().rolling(24).std()
        df1["close_norm"] = close_1h / close_1h.iloc[0]

        # ---------- 4h 趋势 ----------
        df4 = self.df_4h.copy()
        close_4h = df4["close"]
        ema_fast_4h = close_4h.ewm(span=30, adjust=False).mean()
        ema_slow_4h = close_4h.ewm(span=90, adjust=False).mean()
        trend_raw_4h = (ema_fast_4h - ema_slow_4h) / (close_4h + 1e-9)
        df4["trend_raw_4h"] = trend_raw_4h
        df4["trend_dir_4h"] = np.sign(trend_raw_4h)

        abs_t4 = trend_raw_4h.abs()
        lo4, hi4 = abs_t4.quantile(0.1), abs_t4.quantile(0.9)
        span4 = hi4 - lo4 if hi4 > lo4 else 1e-9
        df4["trend_strength_4h"] = ((abs_t4 - lo4) / span4).clip(0, 1)

        # ---------- 15m 特征 ----------
        df15 = self.df_15m.copy()
        close_15 = df15["close"]

        df15["mom_8"] = close_15 / close_15.shift(8) - 1.0
        df15["vol_16"] = close_15.pct_change().rolling(16).std()

        high_15 = df15["high"]
        low_15 = df15["low"]
        df15["hh_40"] = high_15.rolling(40).max().shift(1)
        df15["ll_40"] = low_15.rolling(40).min().shift(1)
        df15["breakout_up"] = close_15 > df15["hh_40"]
        df15["breakout_down"] = close_15 < df15["ll_40"]

        df15["hour_ts"] = df15.index.floor("H")
        g15 = df15.groupby("hour_ts")
        df15_agg = pd.DataFrame({
            "mom_8_1h": g15["mom_8"].mean(),
            "vol_16_1h": g15["vol_16"].mean(),
            "breakout_up_1h": g15["breakout_up"].any(),
            "breakout_down_1h": g15["breakout_down"].any(),
        })

        # ---------- 将 4h 特征对齐到 1h ----------
        df1_reset = df1.reset_index().rename(columns={df1.index.name or "index": "ts"})
        df4_reset = df4[["trend_raw_4h", "trend_dir_4h", "trend_strength_4h"]].reset_index().rename(
            columns={df4.index.name or "index": "ts"}
        )

        merged_4h = pd.merge_asof(
            df1_reset,
            df4_reset,
            on="ts",
            direction="backward",
        ).set_index("ts")

        # ---------- 将 15m 聚合特征对齐到 1h ----------
        df15_agg = df15_agg.sort_index()
        merged_4h_15 = merged_4h.join(df15_agg, how="left")

        merged_4h_15["breakout_up_1h"] = merged_4h_15["breakout_up_1h"].fillna(False)
        merged_4h_15["breakout_down_1h"] = merged_4h_15["breakout_down_1h"].fillna(False)

        # ---------- Regime 判定 ----------
        trend_proxy = (merged_4h_15["ema_fast_ratio"] - merged_4h_15["ema_slow_ratio"]).abs()
        vol_proxy = merged_4h_15["vol_24"].fillna(0)

        t_norm = _zscore(trend_proxy.fillna(0), window=200).clip(-3, 3)
        v_norm = _zscore(vol_proxy, window=200).clip(-3, 3)

        regime = np.zeros(len(merged_4h_15), dtype=int)
        regime[(t_norm > 0.5) & (v_norm > -0.5)] = 1
        regime[(t_norm > 1.0) & (v_norm < 1.5)] = 2

        merged_4h_15["regime"] = regime
        merged_4h_15["trend_proxy"] = trend_proxy
        merged_4h_15["vol_proxy"] = vol_proxy

        # ---------- 清理缺失 ----------
        df_state = merged_4h_15.dropna().copy()

        self.df_state = df_state
        if self.start_index < 20:
            self.start_index = 20
        if self.start_index >= len(self.df_state) - 20:
            self.start_index = max(20, len(self.df_state) // 3)

        tmp_state = self._build_state_vector(
            self.start_index,
            equity=INITIAL_CAPITAL,
            cash=INITIAL_CAPITAL,
            position=Position(0, 0.0, 0.0, 0.0),
        )
        self.state_dim = tmp_state.shape[0]

    # ========= 公共接口 =========

    def reset(self) -> np.ndarray:
        """重置环境，返回初始观察值。"""
        self.done = False
        self.cash = INITIAL_CAPITAL
        self.equity = INITIAL_CAPITAL
        self.prev_equity = INITIAL_CAPITAL
        self.position = Position(side=0, leverage=0.0, entry_price=0.0, size=0.0)
        self.steps_in_episode = 0

        self.current_step = self.start_index
        self.initial_price = float(self.df_state["close"].iloc[self.current_step])

        obs = self._get_state()
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步交易。

        :param action: 0: hold, 1: open long, 2: open short, 3: close
        :return: (obs, reward, done, info)
        """
        if self.done:
            raise RuntimeError("请先调用 reset() 再继续 step。")

        if action not in (0, 1, 2, 3):
            raise ValueError(f"非法动作: {action}")

        # 当前价格（用于开平仓）
        price = float(self.df_state["close"].iloc[self.current_step])

        # 先执行交易动作（可能改变 cash / position / equity）
        self._apply_action(action, price)

        # 记录动作后、价格未变时的权益
        self.prev_equity = self.equity

        # 时间前进一格
        self.current_step += 1
        self.steps_in_episode += 1

        # 下一时刻价格，进行逐小时 MTM
        next_price = float(self.df_state["close"].iloc[self.current_step])
        self._mark_to_market(next_price)

        # ===== Reward Shaping 2.0 =====
        # 1) 核心奖励：本步权益变动（与实际赚钱挂钩）
        delta_equity = self.equity - self.prev_equity
        reward_pnl = delta_equity / INITIAL_CAPITAL

        # 2) 时间惩罚：鼓励高质量决策，避免无意义的长时间僵持
        time_penalty = 0.00001

        # 3) 持仓敞口惩罚：仓位越大，惩罚越高，促使 AI 学会权衡收益和风险
        pos = self.position
        if pos.size > 0:
            position_value = abs(pos.size * next_price)
            if self.equity > 0:
                exposure_ratio = position_value / self.equity
            else:
                exposure_ratio = 5.0
        else:
            exposure_ratio = 0.0
        exposure_penalty = 0.00002 * exposure_ratio

        reward = reward_pnl - time_penalty - exposure_penalty

        # 4) 对 reward 做裁剪，防止极端值
        reward = float(np.clip(reward, -0.05, 0.05))

        # 终止条件
        if self.current_step >= len(self.df_state) - 2:
            self.done = True
        if self.max_steps is not None and self.steps_in_episode >= self.max_steps:
            self.done = True
        if self.equity <= INITIAL_CAPITAL * 0.1:
            # 模拟严重亏损/爆仓
            self.done = True

        obs = self._get_state()
        info = {
            "equity": self.equity,
            "prev_equity": self.prev_equity,
            "cash": self.cash,
            "price": next_price,
            "step": self.steps_in_episode,
            "position_side": self.position.side,
            "position_leverage": self.position.leverage,
            "position_size": self.position.size,
            "reward_pnl": reward_pnl,
            "exposure_penalty": exposure_penalty,
            "time_penalty": time_penalty,
        }
        return obs, float(reward), self.done, info

    def action_space_sample(self) -> int:
        """简单随机动作，用于调试环境行为。"""
        return int(np.random.randint(0, self.action_space_n))

    # ========= 内部交易逻辑 =========

    def _apply_action(self, action: int, price: float) -> None:
        pos = self.position

        # HOLD
        if action == 0:
            return

        # 平仓
        if action == 3 and pos.side != 0:
            self._close_position(price)
            return

        # 开多（空仓时）
        if action == 1 and pos.side == 0:
            self._open_position(side=1, price=price)
            return

        # 开空（空仓时）
        if action == 2 and pos.side == 0:
            self._open_position(side=-1, price=price)
            return

        # 其它不合法的组合，视为 HOLD
        return

    def _open_position(self, side: int, price: float) -> None:
        """
        以固定目标杠杆开仓（暂时固定为 2 倍）。
        使用 cash 作为基础保证金，采用简单 Cross 模型：
            notional = cash * leverage
        """
        if self.cash <= 0:
            return

        target_leverage = 2.0
        notional = self.cash * target_leverage
        size = notional / price

        # 交易手续费
        fee = notional * TRADING_FEE_RATE
        slippage_price = price * (1 + SLIPPAGE_RATE * side)

        # 如果手续费过大，会导致 cash 迅速归零，这里做个防护
        if fee >= self.cash * 0.5:
            return

        self.cash -= fee

        self.position = Position(
            side=side,
            leverage=target_leverage,
            entry_price=slippage_price,
            size=size,
        )

        # 开仓后权益 = 现金 + 未实现盈亏（此刻认为 0）
        self.equity = self.cash

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

        self.cash += pnl
        self.cash -= fee

        self.position = Position(side=0, leverage=0.0, entry_price=0.0, size=0.0)

        # 平仓后权益 = 全部现金
        self.equity = self.cash

    def _mark_to_market(self, price: float) -> None:
        """
        逐小时 MTM：根据当前价格更新未实现盈亏和总权益。
        """
        pos = self.position
        if pos.side == 0 or pos.size <= 0:
            self.equity = self.cash
            return

        notional_entry = pos.entry_price * pos.size
        notional_now = price * pos.size

        if pos.side > 0:
            pnl_unrealized = notional_now - notional_entry
        else:
            pnl_unrealized = notional_entry - notional_now

        self.equity = self.cash + pnl_unrealized

    # ========= 状态构建 =========

    def _build_state_vector(
        self,
        idx: int,
        equity: float,
        cash: float,
        position: Position,
    ) -> np.ndarray:
        """
        将某一时刻 idx 的多周期特征 + 账户特征 组合成 state 向量。
        """
        row = self.df_state.iloc[idx]

        # ---- 1h 价格 & 技术特征 ----
        close_norm = float(row["close_norm"])
        ret_1 = float(row["ret_1"])
        ret_6 = float(row["ret_6"])
        ret_24 = float(row["ret_24"])
        ema_fast_ratio = float(row["ema_fast_ratio"])
        ema_slow_ratio = float(row["ema_slow_ratio"])
        atr_ratio = float(row["atr_ratio"])
        vol_24 = float(row["vol_24"])

        # ---- 4h 趋势特征 ----
        trend_raw_4h = float(row["trend_raw_4h"])
        trend_dir_4h = float(row["trend_dir_4h"])
        trend_strength_4h = float(row["trend_strength_4h"])

        # ---- 15m 动量/波动/突破 ----
        mom_8_1h = float(row["mom_8_1h"]) if not np.isnan(row["mom_8_1h"]) else 0.0
        vol_16_1h = float(row["vol_16_1h"]) if not np.isnan(row["vol_16_1h"]) else 0.0
        breakout_up_1h = 1.0 if bool(row["breakout_up_1h"]) else 0.0
        breakout_down_1h = 1.0 if bool(row["breakout_down_1h"]) else 0.0

        # ---- Regime / Proxy ----
        regime = float(row["regime"])
        trend_proxy = float(row["trend_proxy"])
        vol_proxy = float(row["vol_proxy"])

        # ---- 账户与仓位特征 ----
        pos_side = float(position.side)
        pos_leverage_norm = float(position.leverage / 5.0)  # 归一化
        if position.size > 0 and equity > 0:
            position_value = position.size * float(row["close"])
            position_value_ratio = float(position_value / equity)
        else:
            position_value_ratio = 0.0

        unrealized_pnl_ratio = 0.0  # 这里可以在后续版本中显式计算
        equity_ratio = float(equity / INITIAL_CAPITAL)
        cash_ratio = float(cash / INITIAL_CAPITAL)

        state = np.array(
            [
                # 1h price & tech
                close_norm,
                ret_1,
                ret_6,
                ret_24,
                ema_fast_ratio,
                ema_slow_ratio,
                atr_ratio,
                vol_24,
                # 4h trend
                trend_raw_4h,
                trend_dir_4h,
                trend_strength_4h,
                # 15m features
                mom_8_1h,
                vol_16_1h,
                breakout_up_1h,
                breakout_down_1h,
                # regime & proxies
                regime,
                trend_proxy,
                vol_proxy,
                # account & position
                pos_side,
                pos_leverage_norm,
                position_value_ratio,
                unrealized_pnl_ratio,
                equity_ratio,
                cash_ratio,
            ],
            dtype=np.float32,
        )
        return state

    def _get_state(self) -> np.ndarray:
        return self._build_state_vector(
            idx=self.current_step,
            equity=self.equity,
            cash=self.cash,
            position=self.position,
        )


# ======== 简单自测入口 ========

if __name__ == "__main__":
    # 简单 smoke test：随机动作跑一个 episode 看看 reward 是否不再全 0
    env = TradingEnvV22MultiTF(symbol="BTCUSDT", days=365)
    obs = env.reset()
    logger.info(f"初始 obs 维度: {obs.shape}, 值示例前 5 个: {obs[:5]}")

    done = False
    step_count = 0
    total_reward = 0.0

    while not done and step_count < 300:
        action = env.action_space_sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1

    logger.info(
        f"[EnvV22_4] 随机策略跑完 {step_count} 步, 总 reward={total_reward:.4f}, "
        f"最终权益={info.get('equity', None):.2f}"
    )
