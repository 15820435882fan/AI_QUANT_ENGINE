# real_strategies.py
#!/usr/bin/env python3
import pandas as pd
import numpy as np
from typing import Dict, Optional


# =============================
# 工具函数
# =============================

def _ensure_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col]
    return pd.to_numeric(s, errors="coerce")


# =============================
# 1. MACD 策略
# =============================

def macd_signal(df: pd.DataFrame,
                fast: int = 12,
                slow: int = 26,
                signal: int = 9) -> pd.Series:
    close = _ensure_series(df, "close")

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - macd_signal

    prev = hist.shift(1)
    sig = pd.Series(0, index=df.index, dtype=float)
    sig[(prev <= 0) & (hist > 0)] = 1     # 金叉做多
    sig[(prev >= 0) & (hist < 0)] = -1    # 死叉做空
    return sig


# =============================
# 2. EMA/SMA 趋势跟随策略
# =============================

def ema_trend_signal(df: pd.DataFrame,
                     short: int = 20,
                     long: int = 50) -> pd.Series:
    close = _ensure_series(df, "close")
    ema_short = close.ewm(span=short, adjust=False).mean()
    ema_long = close.ewm(span=long, adjust=False).mean()

    sig = pd.Series(0, index=df.index, dtype=float)
    sig[ema_short > ema_long] = 1   # 短期在长期之上 → 多头趋势
    sig[ema_short < ema_long] = -1  # 空头趋势
    return sig


# =============================
# 3. Turtle 通道突破策略
# =============================

def turtle_signal(df: pd.DataFrame,
                  breakout_window: int = 20,
                  exit_window: int = 10) -> pd.Series:
    high = _ensure_series(df, "high")
    low = _ensure_series(df, "low")
    close = _ensure_series(df, "close")

    hh = high.rolling(window=breakout_window, min_periods=1).max()
    ll = low.rolling(window=breakout_window, min_periods=1).min()
    exit_hh = high.rolling(window=exit_window, min_periods=1).max()
    exit_ll = low.rolling(window=exit_window, min_periods=1).min()

    sig = pd.Series(0, index=df.index, dtype=float)

    # 价格突破最近 N 日最高 → 做多
    sig[close > hh.shift(1)] = 1
    # 价格跌破最近 N 日最低 → 做空
    sig[close < ll.shift(1)] = -1

    # 退出信号可以后面拓展（目前由回测引擎的止损/止盈处理）
    return sig


# =============================
# 4. Bollinger 收敛/扩散策略
# =============================

def bollinger_signal(df: pd.DataFrame,
                     window: int = 20,
                     num_std: float = 2.0) -> pd.Series:
    close = _ensure_series(df, "close")
    ma = close.rolling(window=window, min_periods=1).mean()
    std = close.rolling(window=window, min_periods=1).std().fillna(0)

    upper = ma + num_std * std
    lower = ma - num_std * std

    sig = pd.Series(0, index=df.index, dtype=float)

    # 下轨以下超卖 → 博反弹做多
    sig[close < lower] = 1
    # 上轨以上超买 → 博回落做空
    sig[close > upper] = -1

    return sig


# =============================
# 5. 简单 Breakout 策略（区间突破）
# =============================

def breakout_signal(df: pd.DataFrame,
                    lookback: int = 50,
                    threshold: float = 0.01) -> pd.Series:
    close = _ensure_series(df, "close")
    rolling_max = close.rolling(window=lookback, min_periods=1).max()
    rolling_min = close.rolling(window=lookback, min_periods=1).min()

    sig = pd.Series(0, index=df.index, dtype=float)

    # 向上突破最近区间高点一定幅度
    sig[close > rolling_max * (1 + threshold)] = 1
    # 向下跌破最近区间低点一定幅度
    sig[close < rolling_min * (1 - threshold)] = -1

    return sig


# =============================
# 6. 多策略合成（Ensemble）
# =============================

def build_ensemble_signal(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    返回一个综合信号:
      +1 = 多头, -1 = 空头, 0 = 观望
    """
    if weights is None:
        weights = {
            "macd": 1.0,
            "ema": 1.0,
            "turtle": 1.0,
            "boll": 1.0,
            "breakout": 1.0,
        }

    sig_macd = macd_signal(df)
    sig_ema = ema_trend_signal(df)
    sig_turtle = turtle_signal(df)
    sig_boll = bollinger_signal(df)
    sig_break = breakout_signal(df)

    # 对齐索引
    idx = df.index
    total = (
        weights.get("macd", 0) * sig_macd.reindex(idx, fill_value=0) +
        weights.get("ema", 0) * sig_ema.reindex(idx, fill_value=0) +
        weights.get("turtle", 0) * sig_turtle.reindex(idx, fill_value=0) +
        weights.get("boll", 0) * sig_boll.reindex(idx, fill_value=0) +
        weights.get("breakout", 0) * sig_break.reindex(idx, fill_value=0)
    )

    ensemble = pd.Series(0, index=idx, dtype=float)
    ensemble[total > 0] = 1
    ensemble[total < 0] = -1

    return ensemble
