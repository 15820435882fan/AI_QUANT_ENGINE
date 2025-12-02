# v30_teacher_strategies.py
#
# V30 Teacher strategies:
# - Simple but explicit trend-following teacher for BTCUSDT (or other symbols)
# - Generates stateful trade actions and risk labels over a 1h OHLCV series
#
# Actions (discrete):
#   0 = HOLD (no change)
#   1 = OPEN / HOLD LONG
#   2 = OPEN / HOLD SHORT
#   3 = CLOSE (go flat)
#
# Risk levels:
#   0 = low
#   1 = medium
#   2 = high
#
# The goal is not to be perfect, but to be:
#   - interpretable
#   - reasonably trend-following
#   - robust enough to produce supervised learning targets.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TeacherConfig:
    fast_ma: int = 20
    slow_ma: int = 60
    atr_period: int = 14
    atr_stop_mult: float = 2.5
    take_profit_mult: float = 4.0
    min_trend_strength: float = 0.002  # minimal MA slope / price to consider trend
    min_ma_distance: float = 0.002     # minimal MA separation / price


@dataclass
class TeacherState:
    position_side: int = 0     # -1 short, 0 flat, 1 long
    entry_price: float = 0.0
    atr_at_entry: float = 0.0


def compute_basic_indicators(df: pd.DataFrame, cfg: TeacherConfig) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]

    df["ma_fast"] = close.rolling(cfg.fast_ma).mean()
    df["ma_slow"] = close.rolling(cfg.slow_ma).mean()

    df["ma_fast_ratio"] = df["ma_fast"] / close - 1.0
    df["ma_slow_ratio"] = df["ma_slow"] / close - 1.0

    # simple ATR
    high = df["high"]
    low = df["low"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(cfg.atr_period).mean()
    df["atr_ratio"] = df["atr"] / close

    # simple trend slope over slow_ma window
    df["close_shift_slow"] = close.shift(cfg.slow_ma)
    df["trend_slope"] = (close - df["close_shift_slow"]) / (df["close_shift_slow"] + 1e-9)

    # basic volatility proxy
    df["ret_1"] = close.pct_change()
    df["vol_24"] = df["ret_1"].rolling(24).std()

    return df


def classify_risk_level(atr_ratio: pd.Series) -> pd.Series:
    """Classify risk level from ATR ratio via quantiles.

    We compute 33% and 66% quantiles and map:
        atr <= q1  -> low  (0)
        q1 < atr <= q2 -> medium (1)
        atr > q2  -> high (2)
    """
    s = atr_ratio.dropna()
    if len(s) < 10:
        return pd.Series(index=atr_ratio.index, data=1, dtype=int)

    q1 = s.quantile(0.33)
    q2 = s.quantile(0.66)

    def _risk(a: float) -> int:
        if np.isnan(a):
            return 1
        if a <= q1:
            return 0
        if a <= q2:
            return 1
        return 2

    return atr_ratio.apply(_risk).astype(int)


def teacher_trend_follow(df: pd.DataFrame, cfg: Optional[TeacherConfig] = None) -> pd.DataFrame:
    """Generate teacher actions and risk labels on 1h OHLCV data.

    Input df:
        index: datetime
        columns: open, high, low, close, (volume optional)

    Returns a DataFrame with:
        - all original columns
        - indicators
        - action_teacher (int)
        - risk_level (int)
        - position_side_teacher (int)
        - entry_price_teacher (float)
    """
    if cfg is None:
        cfg = TeacherConfig()

    df = compute_basic_indicators(df, cfg)
    df["risk_level"] = classify_risk_level(df["atr_ratio"])

    # initialize outputs
    n = len(df)
    actions = np.zeros(n, dtype=int)
    pos_side_arr = np.zeros(n, dtype=int)
    entry_price_arr = np.zeros(n, dtype=float)

    state = TeacherState()

    for i in range(n):
        row = df.iloc[i]
        price = float(row["close"])
        ma_fast = float(row["ma_fast"])
        ma_slow = float(row["ma_slow"])
        atr = float(row["atr"])
        trend_slope = float(row["trend_slope"])

        action = 0  # default HOLD

        # We require indicators to be valid
        if np.isnan(ma_fast) or np.isnan(ma_slow) or np.isnan(atr) or np.isnan(trend_slope):
            actions[i] = 0
            pos_side_arr[i] = state.position_side
            entry_price_arr[i] = state.entry_price
            continue

        # Trend direction and strength
        ma_distance = (ma_fast - ma_slow) / (price + 1e-9)
        up_trend = (ma_fast > ma_slow) and (ma_distance > cfg.min_ma_distance) and (trend_slope > cfg.min_trend_strength)
        down_trend = (ma_fast < ma_slow) and (ma_distance < -cfg.min_ma_distance) and (trend_slope < -cfg.min_trend_strength)

        # Stop-loss and take-profit levels if in position
        if state.position_side != 0:
            if state.position_side > 0:
                stop_price = state.entry_price - cfg.atr_stop_mult * state.atr_at_entry
                tp_price = state.entry_price + cfg.take_profit_mult * state.atr_at_entry
                if price <= stop_price or price >= tp_price or not up_trend:
                    action = 3  # close long
                    state.position_side = 0
                    state.entry_price = 0.0
                    state.atr_at_entry = 0.0
                else:
                    action = 1  # hold long
            else:
                stop_price = state.entry_price + cfg.atr_stop_mult * state.atr_at_entry
                tp_price = state.entry_price - cfg.take_profit_mult * state.atr_at_entry
                if price >= stop_price or price <= tp_price or not down_trend:
                    action = 3  # close short
                    state.position_side = 0
                    state.entry_price = 0.0
                    state.atr_at_entry = 0.0
                else:
                    action = 2  # hold short
        else:
            # flat: decide whether to open new position
            if up_trend:
                action = 1
                state.position_side = 1
                state.entry_price = price
                state.atr_at_entry = atr
            elif down_trend:
                action = 2
                state.position_side = -1
                state.entry_price = price
                state.atr_at_entry = atr
            else:
                action = 0

        actions[i] = action
        pos_side_arr[i] = state.position_side
        entry_price_arr[i] = state.entry_price

    df["action_teacher"] = actions
    df["position_side_teacher"] = pos_side_arr
    df["entry_price_teacher"] = entry_price_arr

    # drop early rows where indicators are not ready
    df = df.dropna(subset=["ma_fast", "ma_slow", "atr", "trend_slope"]).copy()

    return df
