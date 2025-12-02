"""
trading_env_v22_9.py

V22_9: 多周期交易环境 + Train/Valid/Test 数据切分版
----------------------------------------------------
在 V22_5 的基础上增加：

1. 数据按时间顺序切分为三段：
   - segment="train":  前 train_ratio 部分
   - segment="valid":  中间 valid_ratio 部分
   - segment="test":   剩余部分

2. 其它逻辑保持与 V22_5 一致：
   - 多周期 (4h / 1h / 15m) 特征
   - MTM 账户权益
   - Reward Shaping 3.0（顺势奖励、逆势惩罚、回撤惩罚等）

该环境专为 V22_6 PPO 训练器使用，支持：
    - 训练集环境：强化学习训练
    - 验证集环境：Early Stop / 模型选择
    - 测试集环境：最终评估
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

INITIAL_CAPITAL = 10_000.0
TRADING_FEE_RATE = 0.0005  # 单边手续费
SLIPPAGE_RATE = 0.0002     # 简单滑点

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
    side: int        # -1: short, 0: flat, 1: long
    leverage: float
    entry_price: float
    size: float      # 持仓数量（张数，notional = price * size）


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


class TradingEnvV22MultiTFV9:
    """
    强化学习交易环境（单标的，多周期状态，Reward Shaping 3.0 版 + 数据切分）

    segment:
        - "train": 使用前 train_ratio 的数据
        - "valid": 使用中间 valid_ratio 的数据
        - "test" : 使用剩余部分
    """

    def __init__(
        self,
        symbol: str,
        days: int = 365,
        start_index: int = 300,
        max_steps: Optional[int] = None,
        segment: str = "train",
        train_ratio: float = 0.7,
        valid_ratio: float = 0.2,
    ):
        assert segment in ("train", "valid", "test"), f"非法 segment: {segment}"
        assert 0 < train_ratio < 1 and 0 < valid_ratio < 1 and train_ratio + valid_ratio < 1

        self.symbol = symbol
        self.days = days
        self.start_index = start_index
        self.max_steps = max_steps
        self.segment = segment
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio

        # 原始多周期数据
        self.df_1h: Optional[pd.DataFrame] = None
        self.df_15m: Optional[pd.DataFrame] = None
        self.df_4h: Optional[pd.DataFrame] = None

        # 对齐后完整的状态表 & 当前 segment 子集
        self.df_state_full: Optional[pd.DataFrame] = None
        self.df_state: Optional[pd.DataFrame] = None

        # 账户 & 仓位
        self.current_step: int = 0
        self.cash: float = INITIAL_CAPITAL
        self.equity: float = INITIAL_CAPITAL
        self.initial_price: float = 0.0
        self.position: Position = Position(side=0, leverage=0.0, entry_price=0.0, size=0.0)
        self.done: bool = False

        self.prev_equity: float = INITIAL_CAPITAL
        self.steps_in_episode: int = 0
        self.equity_peak: float = INITIAL_CAPITAL
        self.last_realized_pnl: float = 0.0

        # 加载多周期数据 + 状态 + 切片
        self._load_data_multitf()
        self._build_multitf_state()
        self._apply_segment_split()

        self.action_space_n = 4

        # 预先构建一个 state 模板，获取维度
        tmp_state = self._build_state_vector(
            idx=0,
            equity=INITIAL_CAPITAL,
            cash=INITIAL_CAPITAL,
            position=Position(0, 0.0, 0.0, 0.0),
            unrealized_pnl_ratio=0.0,
        )
        self.state_dim = tmp_state.shape[0]

        logger.info(
            f"[EnvV22_9] segment={self.segment}, usable_len={len(self.df_state)}, "
            f"state_dim={self.state_dim}"
        )

    # ========= 多周期数据 & 状态 =========

    def _load_data_multitf(self) -> None:
        """加载多周期 K 线数据 (1h / 15m / 4h)。优先通过 V22_9 本地数据引擎
        从 feather / CSV 统一读取，保持与 V22_6 相同的多周期结构。"""
        from local_data_engine_v22_9 import LocalDataEngineV22_9

        logger.info(f"[EnvV22_9] 加载多周期数据: {self.symbol}, days={self.days}")

        engine = LocalDataEngineV22_9(
            feather_dir="data/feather",
            csv_base_dir="data",
            exchange="binance",
        )

        raw_1h = engine.load_klines(self.symbol, "1h", self.days)
        raw_15m = engine.load_klines(self.symbol, "15m", self.days)
        raw_4h = engine.load_klines(self.symbol, "4h", self.days + 60)

        self.df_1h = _ensure_ohlc(raw_1h)
        self.df_15m = _ensure_ohlc(raw_15m)
        self.df_4h = _ensure_ohlc(raw_4h)

    def _build_multitf_state(self) -> None:
        df1 = self.df_1h.copy()
        close_1h = df1["close"]

        # ----- 1h 特征 -----
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

        # ----- 4h 趋势 -----
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

        # ----- 15m 动量/突破 -----
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

        # ----- 对齐 4h 到 1h -----
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

        # ----- 对齐 15m 聚合到 1h -----
        df15_agg = df15_agg.sort_index()
        merged_4h_15 = merged_4h.join(df15_agg, how="left")

        merged_4h_15["breakout_up_1h"] = merged_4h_15["breakout_up_1h"].fillna(False)
        merged_4h_15["breakout_down_1h"] = merged_4h_15["breakout_down_1h"].fillna(False)

        # ----- Regime 判定 -----
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

        df_state = merged_4h_15.dropna().copy()
        self.df_state_full = df_state

    def _apply_segment_split(self) -> None:
        """根据 segment / ratio 将 df_state_full 切分为 train/valid/test 段。"""
        df = self.df_state_full
        n = len(df)
        if n < 100:
            raise ValueError(f"可用状态数据太少: {n}")

        train_end = int(n * self.train_ratio)
        valid_end = int(n * (self.train_ratio + self.valid_ratio))

        # 防止边界问题
        train_end = max(train_end, 30)
        valid_end = max(valid_end, train_end + 10)
        valid_end = min(valid_end, n - 20)

        if self.segment == "train":
            seg_df = df.iloc[:train_end].copy()
        elif self.segment == "valid":
            seg_df = df.iloc[train_end:valid_end].copy()
        else:
            seg_df = df.iloc[valid_end:].copy()

        if len(seg_df) < 50:
            raise ValueError(f"segment={self.segment} 数据太短: {len(seg_df)}")

        self.df_state = seg_df.reset_index(drop=True)

        # 调整 start_index 相对于分段后的 df_state
        if self.start_index < 20:
            self.start_index = 20
        if self.start_index >= len(self.df_state) - 20:
            self.start_index = max(20, len(self.df_state) // 3)

    # ========= 接口 =========

    def reset(self) -> np.ndarray:
        self.done = False
        self.cash = INITIAL_CAPITAL
        self.equity = INITIAL_CAPITAL
        self.prev_equity = INITIAL_CAPITAL
        self.equity_peak = INITIAL_CAPITAL
        self.position = Position(side=0, leverage=0.0, entry_price=0.0, size=0.0)
        self.steps_in_episode = 0
        self.last_realized_pnl = 0.0

        self.current_step = self.start_index
        self.initial_price = float(self.df_state["close"].iloc[self.current_step])

        obs = self._get_state(unrealized_pnl=0.0)
        return obs

    def step(self, action: int):
        if self.done:
            raise RuntimeError("环境已结束，请先 reset()。")
        if action not in (0, 1, 2, 3):
            raise ValueError(f"非法动作: {action}")

        self.last_realized_pnl = 0.0

        price = float(self.df_state["close"].iloc[self.current_step])
        self._apply_action(action, price)
        self.prev_equity = self.equity

        # 时间前进一步
        self.current_step += 1
        self.steps_in_episode += 1
        next_price = float(self.df_state["close"].iloc[self.current_step])

        # Mark-to-Market 未实现盈亏
        unrealized_pnl = self._mark_to_market(next_price)

        # 记录 equity 峰值
        if self.equity > self.equity_peak:
            self.equity_peak = self.equity

        # ===== Reward Shaping 3.0 =====
        # ===== Reward Shaping 4.0: 趋势强化版 =====
        row = self.df_state.iloc[self.current_step]
        trend_dir_4h = float(row["trend_dir_4h"])
        trend_strength_4h = float(row["trend_strength_4h"])
        regime = float(row["regime"])

        # 1) 基础 PnL 回报：保持与 V22_6 一致
        delta_equity = self.equity - self.prev_equity
        pnl_reward = delta_equity / INITIAL_CAPITAL

        pos = self.position
        trend_reward = 0.0
        momentum_reward = 0.0

        if pos.side != 0:
            # 正：顺势，负：逆势
            align = pos.side * trend_dir_4h

            # 2) 趋势强度加权（放大强趋势区间的顺势收益）
            if align > 0 and trend_strength_4h > 0:
                # regime == 2 视为强趋势，给额外加成
                regime_factor = 1.0 + 0.5 * (regime == 2)
                # 使用 trend_strength^2，让强趋势段更突出、弱趋势影响更小
                trend_reward = 0.0015 * (trend_strength_4h ** 2) * regime_factor

            # 3) 短期动量一致性奖励：价涨持多 / 价跌持空
            price_change = (next_price - price) / max(price, 1e-9)
            if pos.side * price_change > 0:
                # 越是顺着短期动量，越给一点小奖励
                momentum_reward = 0.0005 * abs(price_change) * (1.0 + trend_strength_4h)

        # 4) 平仓奖励：鼓励在有真实盈亏时主动了结
        close_reward = 0.0
        if action == 3 and abs(self.last_realized_pnl) > 1e-6:
            close_reward = 0.05 * (self.last_realized_pnl / INITIAL_CAPITAL)

        # 5) 时间 & 曝光惩罚：略微减轻时间惩罚，防止过度打压持仓
        time_penalty = 0.000005  # 比 3.0 略低

        if pos.size > 0 and self.equity > 0:
            position_value = abs(pos.size * next_price)
            exposure_ratio = position_value / self.equity
        else:
            exposure_ratio = 0.0
        exposure_penalty = 0.00001 * exposure_ratio

        # 6) 逆趋势惩罚：在明显趋势+较强强度下逆势，加大惩罚
        anti_trend_penalty = 0.0
        if pos.side != 0:
            align = pos.side * trend_dir_4h
            if align < 0 and trend_strength_4h > 0.3:
                anti_trend_penalty = 0.0025 * trend_strength_4h  # 略强于 3.0

        # 7) 深度回撤惩罚：保持原有逻辑
        dd_ratio = (self.equity - self.equity_peak) / (self.equity_peak + 1e-9)
        drawdown_penalty = 0.0
        if dd_ratio < -0.4:
            drawdown_penalty = -0.01

        reward = (
            pnl_reward
            + trend_reward
            + momentum_reward
            + close_reward
            - time_penalty
            - exposure_penalty
            - anti_trend_penalty
            + drawdown_penalty
        )
        reward = float(np.clip(reward, -0.1, 0.1))


        # 终止条件
        if self.current_step >= len(self.df_state) - 2:
            self.done = True
        if self.max_steps is not None and self.steps_in_episode >= self.max_steps:
            self.done = True
        if self.equity <= INITIAL_CAPITAL * 0.1:
            self.done = True

        obs = self._get_state(unrealized_pnl=unrealized_pnl)
        info = {
            "equity": self.equity,
            "prev_equity": self.prev_equity,
            "cash": self.cash,
            "price": next_price,
            "step": self.steps_in_episode,
            "position_side": self.position.side,
            "position_leverage": self.position.leverage,
            "position_size": self.position.size,
            "reward_pnl": pnl_reward,
            "trend_reward": trend_reward,
            "close_reward": close_reward,
            "exposure_penalty": exposure_penalty,
            "time_penalty": time_penalty,
            "anti_trend_penalty": anti_trend_penalty,
            "drawdown_penalty": drawdown_penalty,
            "dd_ratio": dd_ratio,
        }
        return obs, reward, self.done, info

    def action_space_sample(self) -> int:
        return int(np.random.randint(0, self.action_space_n))

    # ========= 内部交易逻辑 =========

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
        if self.cash <= 0:
            return

        target_leverage = 2.0
        notional = self.cash * target_leverage
        size = notional / price

        fee = notional * TRADING_FEE_RATE
        slippage_price = price * (1 + SLIPPAGE_RATE * side)

        if fee >= self.cash * 0.5:
            return

        self.cash -= fee

        self.position = Position(
            side=side,
            leverage=target_leverage,
            entry_price=slippage_price,
            size=size,
        )

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
        self.equity = self.cash

        self.last_realized_pnl = pnl - fee

    def _mark_to_market(self, price: float) -> float:
        pos = self.position
        if pos.side == 0 or pos.size <= 0:
            self.equity = self.cash
            return 0.0

        notional_entry = pos.entry_price * pos.size
        notional_now = price * pos.size

        if pos.side > 0:
            pnl_unrealized = notional_now - notional_entry
        else:
            pnl_unrealized = notional_entry - notional_now

        self.equity = self.cash + pnl_unrealized
        return pnl_unrealized

    # ========= 状态构建 =========

    def _build_state_vector(
        self,
        idx: int,
        equity: float,
        cash: float,
        position: Position,
        unrealized_pnl_ratio: float,
    ) -> np.ndarray:
        row = self.df_state.iloc[idx]

        close_norm = float(row["close_norm"])
        ret_1 = float(row["ret_1"])
        ret_6 = float(row["ret_6"])
        ret_24 = float(row["ret_24"])
        ema_fast_ratio = float(row["ema_fast_ratio"])
        ema_slow_ratio = float(row["ema_slow_ratio"])
        atr_ratio = float(row["atr_ratio"])
        vol_24 = float(row["vol_24"])

        trend_raw_4h = float(row["trend_raw_4h"])
        trend_dir_4h = float(row["trend_dir_4h"])
        trend_strength_4h = float(row["trend_strength_4h"])

        mom_8_1h = float(row["mom_8_1h"]) if not np.isnan(row["mom_8_1h"]) else 0.0
        vol_16_1h = float(row["vol_16_1h"]) if not np.isnan(row["vol_16_1h"]) else 0.0
        breakout_up_1h = 1.0 if bool(row["breakout_up_1h"]) else 0.0
        breakout_down_1h = 1.0 if bool(row["breakout_down_1h"]) else 0.0

        regime = float(row["regime"])
        trend_proxy = float(row["trend_proxy"])
        vol_proxy = float(row["vol_proxy"])

        pos_side = float(position.side)
        pos_leverage_norm = float(position.leverage / 5.0)
        if position.size > 0 and equity > 0:
            position_value = position.size * float(row["close"])
            position_value_ratio = float(position_value / equity)
        else:
            position_value_ratio = 0.0

        equity_ratio = float(equity / INITIAL_CAPITAL)
        cash_ratio = float(cash / INITIAL_CAPITAL)

        state = np.array(
            [
                close_norm,
                ret_1,
                ret_6,
                ret_24,
                ema_fast_ratio,
                ema_slow_ratio,
                atr_ratio,
                vol_24,
                trend_raw_4h,
                trend_dir_4h,
                trend_strength_4h,
                mom_8_1h,
                vol_16_1h,
                breakout_up_1h,
                breakout_down_1h,
                regime,
                trend_proxy,
                vol_proxy,
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

    def _get_state(self, unrealized_pnl: float) -> np.ndarray:
        unrealized_pnl_ratio = unrealized_pnl / (INITIAL_CAPITAL + 1e-9)
        return self._build_state_vector(
            idx=self.current_step,
            equity=self.equity,
            cash=self.cash,
            position=self.position,
            unrealized_pnl_ratio=unrealized_pnl_ratio,
        )


if __name__ == "__main__":
    # 简单双随机策略自检
    env = TradingEnvV22MultiTF(symbol="BTCUSDT", days=365, segment="train")
    obs = env.reset()
    logger.info(f"[EnvV22_9] 初始 obs 维度: {obs.shape}, 示例前 5 个: {obs[:5]}")
    total_r = 0.0
    steps = 0
    done = False
    while not done and steps < 300:
        a = env.action_space_sample()
        obs, r, done, info = env.step(a)
        total_r += r
        steps += 1
    logger.info(
        f"[EnvV22_9] 随机策略测试结束: 步数={steps}, 总reward={total_r:.4f}, 最终权益={info.get('equity', None):.2f}"
    )