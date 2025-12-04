
# -*- coding: utf-8 -*-
"""
v31_core_v31_1.py

V31_1 核心策略内核（共用指标 + 入场 + 止损止盈计算）
- 提供给实盘引擎 / 其他回测脚本复用
- 指标和信号逻辑与 v31_rule_trend_system_v31_1 中保持一致（复制而来）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Tuple, Dict, Any
import numpy as np
import pandas as pd

Side = Literal["FLAT", "LONG", "SHORT"]
TradeType = Literal["TREND"]


# ===========================
# 抽水 + 复利账户模型（与 V31_1 一致）
# ===========================
class WaterfallAccountV31_1:
    """
    抽水 + 复利账户模型（与 v31_rule_trend_system_v31_1 中保持一致）
    """
    def __init__(
        self,
        initial_capital: float = 10_000.0,
        withdraw_rate: float = 0.10,
        growth_threshold: float = 0.01,
        enable_waterfall: bool = True,
    ):
        self.initial_capital = initial_capital
        self.trading_capital = initial_capital
        self.profit_pool = 0.0

        self.withdraw_rate = withdraw_rate
        self.growth_threshold = growth_threshold
        self.enable_waterfall = enable_waterfall

        self.last_high = initial_capital
        self.history = []

    def risk_capital(self) -> float:
        return self.trading_capital

    def total_equity(self) -> float:
        return self.trading_capital + self.profit_pool

    def apply_pnl(self, pnl: float, timestamp: pd.Timestamp):
        before = self.trading_capital
        self.trading_capital += pnl

        if not self.enable_waterfall:
            return

        if self.trading_capital <= self.last_high:
            return

        growth = self.trading_capital - self.last_high
        growth_pct = growth / self.last_high

        if growth_pct < self.growth_threshold:
            return

        withdraw_amount = growth * self.withdraw_rate
        self.trading_capital -= withdraw_amount
        self.profit_pool += withdraw_amount

        self.last_high = self.trading_capital

        record = {
            "timestamp": timestamp,
            "growth_pct": float(growth_pct),
            "growth_amount": float(growth),
            "withdrawn": float(withdraw_amount),
            "trading_before": float(before),
            "trading_after": float(self.trading_capital),
            "profit_pool": float(self.profit_pool),
        }
        self.history.append(record)


# ===========================
# 策略配置（精简版，仅保留策略和风控相关字段）
# ===========================
@dataclass
class V31CoreConfig:
    # 基本
    symbol: str = "BTCUSDT"
    days: int = 365

    # 趋势过滤参数（1H / 4H）
    hma_period_1h: int = 20
    hma_period_4h: int = 20
    adx_period_1h: int = 14
    adx_period_4h: int = 14
    adx_strong_th: float = 25.0
    adx_normal_th: float = 20.0

    # 执行周期（5m）指标
    boll_window_5m: int = 20
    boll_k_5m: float = 2.0
    ema_fast_5m: int = 10
    ema_slow_5m: int = 30
    macd_fast_5m: int = 12
    macd_slow_5m: int = 26
    macd_signal_5m: int = 9

    # ATR 止损（1H）
    atr_period_1h: int = 14
    min_sl_pct: float = 0.004

    # 风控
    risk_per_trade: float = 0.01
    leverage: float = 3.0
    pos_cap_strong: float = 1.0
    pos_cap_normal: float = 0.7

    max_consecutive_losses: int = 5
    atr_spike_threshold: float = 2.0

    # 持仓时间（bar 数）
    max_bars_strong: int = 96
    max_bars_normal: int = 72

    # 手续费 & 滑点
    fee_rate: float = 0.0004
    slippage: float = 0.0002

    # RR & ATR 倍数（与 V31_1 一致，可调）
    rr_strong: float = 4.0
    rr_normal: float = 3.0
    sl_mult_strong: float = 3.5
    sl_mult_normal: float = 3.0


# ===========================
# 工具指标函数（复制自 V31_1）
# ===========================
def wma(series: pd.Series, period: int) -> pd.Series:
    if period <= 0:
        return series * np.nan
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(),
        raw=True,
    )


def hma(series: pd.Series, period: int) -> pd.Series:
    if period < 2:
        return series
    half = wma(series, period // 2)
    full = wma(series, period)
    raw = 2 * half - full
    hma_period = int(np.sqrt(period))
    if hma_period < 1:
        hma_period = 1
    return wma(raw, hma_period)


def adx(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    plus_dm = high - prev_high
    minus_dm = prev_low - low

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_val = tr.rolling(period).mean()

    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(period).sum() / (atr_val + 1e-12))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(period).sum() / (atr_val + 1e-12))

    dx = (plus_di - minus_di).abs() / ((plus_di + minus_di).abs() + 1e-12) * 100
    adx_val = dx.rolling(period).mean()
    return adx_val


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.rolling(period).mean()
    return atr_val


# ===========================
# 多周期指标构建（5m → 1H / 4H）
# ===========================
def build_multi_tf_indicators(df_5m: pd.DataFrame, cfg: V31CoreConfig) -> Dict[str, pd.DataFrame]:
    """
    复制自 V31_1._calc_indicators，只是改为函数形式，返回字典：
    {
        "df_5m": df5_with_indicators,
        "df_1h": df_1h,
        "df_4h": df_4h,
    }
    """
    df5 = df_5m.copy()
    df5 = df5.sort_index()
    close5 = df5["close"]

    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    df_1h = df5.resample("1H").agg(ohlc).dropna()
    df_4h = df5.resample("4H").agg(ohlc).dropna()

    # HMA
    df_1h["hma"] = hma(df_1h["close"], cfg.hma_period_1h)
    df_4h["hma"] = hma(df_4h["close"], cfg.hma_period_4h)

    df_1h["hma_dir"] = np.where(df_1h["close"] > df_1h["hma"], 1, -1)
    df_4h["hma_dir"] = np.where(df_4h["close"] > df_4h["hma"], 1, -1)

    # ADX
    df_1h["adx"] = adx(df_1h, cfg.adx_period_1h)
    df_4h["adx"] = adx(df_4h, cfg.adx_period_4h)

    df_4h_to_1h = df_4h[["hma_dir", "adx"]].reindex(df_1h.index, method="ffill")
    df_1h["hma_dir_4h"] = df_4h_to_1h["hma_dir"]
    df_1h["adx_4h"] = df_4h_to_1h["adx"]

    same_sign = (df_1h["hma_dir"] * df_1h["hma_dir_4h"]) > 0
    trend_dir_1h = np.where(same_sign, df_1h["hma_dir"], 0)

    adx1 = df_1h["adx"].fillna(0.0)
    adx4 = df_1h["adx_4h"].fillna(0.0)
    adx_combined = (adx1 + adx4) / 2.0

    strong = adx_combined >= cfg.adx_strong_th
    normal = (adx_combined >= cfg.adx_normal_th) & (adx_combined < cfg.adx_strong_th)
    trend_strength_1h = np.zeros(len(df_1h), dtype=int)
    trend_strength_1h[normal] = 1
    trend_strength_1h[strong] = 2
    trend_strength_1h[trend_dir_1h == 0] = 0

    df_1h["trend_dir"] = trend_dir_1h
    df_1h["trend_strength"] = trend_strength_1h
    df_1h["adx_combined"] = adx_combined

    # ATR + ATR spike
    df_1h["atr_1h"] = atr(df_1h, cfg.atr_period_1h)
    df_1h["atr_ma"] = df_1h["atr_1h"].rolling(20).mean()
    df_1h["atr_spike"] = df_1h["atr_1h"] / (df_1h["atr_ma"] + 1e-12)

    # 对齐到 5m
    df5["trend_dir"] = df_1h["trend_dir"].reindex(df5.index, method="ffill").fillna(0).astype(int)
    df5["trend_strength"] = df_1h["trend_strength"].reindex(df5.index, method="ffill").fillna(0).astype(int)
    df5["hma_dir_1h"] = df_1h["hma_dir"].reindex(df5.index, method="ffill").fillna(0).astype(int)
    df5["hma_dir_4h"] = df_1h["hma_dir_4h"].reindex(df5.index, method="ffill").fillna(0).astype(int)
    df5["adx_1h"] = df_1h["adx"].reindex(df5.index, method="ffill").fillna(0.0)
    df5["adx_4h"] = df_1h["adx_4h"].reindex(df5.index, method="ffill").fillna(0.0)
    df5["adx_combined"] = df_1h["adx_combined"].reindex(df5.index, method="ffill").fillna(0.0)
    df5["atr_1h"] = df_1h["atr_1h"].reindex(df5.index, method="ffill")
    df5["atr_spike"] = df_1h["atr_spike"].reindex(df5.index, method="ffill")

    # BOLL, EMA, MACD
    mid5 = close5.rolling(cfg.boll_window_5m).mean()
    std5 = close5.rolling(cfg.boll_window_5m).std()
    up5 = mid5 + cfg.boll_k_5m * std5
    low5 = mid5 - cfg.boll_k_5m * std5
    df5["boll_mid_5m"] = mid5
    df5["boll_up_5m"] = up5
    df5["boll_low_5m"] = low5

    ema_fast_5 = close5.ewm(span=cfg.ema_fast_5m, adjust=False).mean()
    ema_slow_5 = close5.ewm(span=cfg.ema_slow_5m, adjust=False).mean()
    df5["ema_fast_5m"] = ema_fast_5
    df5["ema_slow_5m"] = ema_slow_5

    ema_fast_macd5 = close5.ewm(span=cfg.macd_fast_5m, adjust=False).mean()
    ema_slow_macd5 = close5.ewm(span=cfg.macd_slow_5m, adjust=False).mean()
    macd5 = ema_fast_macd5 - ema_slow_macd5
    macd5_sig = macd5.ewm(span=cfg.macd_signal_5m, adjust=False).mean()
    df5["macd_hist_5m"] = macd5 - macd5_sig

    df5["is_15m_bar"] = (df5.index.minute % 15 == 0)
    df5 = df5.dropna().copy()

    return {"df_5m": df5, "df_1h": df_1h, "df_4h": df_4h}


# ===========================
# 入场信号（复制 V31_1._entry_signal）
# ===========================
def entry_signal_v31(
    cfg: V31CoreConfig,
    consecutive_losses: int,
    ts: pd.Timestamp,
    row: pd.Series,
    prev_row: pd.Series,
) -> Tuple[Side, Optional[TradeType]]:
    trend_dir = int(row["trend_dir"])
    trend_strength = int(row["trend_strength"])

    # 连续亏损保护
    if consecutive_losses >= cfg.max_consecutive_losses:
        return "FLAT", None

    # 极端波动过滤
    if row.get("atr_spike", 1.0) > cfg.atr_spike_threshold:
        return "FLAT", None

    if trend_strength <= 0 or trend_dir == 0:
        return "FLAT", None

    if not bool(row["is_15m_bar"]):
        return "FLAT", None

    close = row["close"]
    mid5 = row["boll_mid_5m"]
    up5 = row["boll_up_5m"]
    low5 = row["boll_low_5m"]

    if mid5 <= 0 or up5 <= 0 or low5 <= 0:
        return "FLAT", None
    rng = up5 - low5
    if rng <= 0:
        return "FLAT", None

    boll_pos = (close - low5) / rng
    ema_fast = row["ema_fast_5m"]
    ema_slow = row["ema_slow_5m"]
    ema_fast_prev = prev_row["ema_fast_5m"]
    ema_slow_prev = prev_row["ema_slow_5m"]

    macd_5 = row["macd_hist_5m"]
    macd_5_prev = prev_row["macd_hist_5m"]

    # 多头
    if trend_dir > 0:
        cond_price = 0.3 <= boll_pos <= 0.85
        cond_trigger = (
            (ema_fast_prev <= ema_slow_prev and ema_fast > ema_slow)
            or (macd_5_prev <= 0 and macd_5 > 0)
        )
        if cond_price and cond_trigger:
            return "LONG", "TREND"

    # 空头
    if trend_dir < 0:
        cond_price = 0.15 <= boll_pos <= 0.7
        cond_trigger = (
            (ema_fast_prev >= ema_slow_prev and ema_fast < ema_slow)
            or (macd_5_prev >= 0 and macd_5 < 0)
        )
        if cond_price and cond_trigger:
            return "SHORT", "TREND"

    return "FLAT", None


# ===========================
# 止损 / 止盈 / 仓位（复制 V31_1._compute_sl_tp_notional）
# ===========================
def compute_sl_tp_notional_v31(
    cfg: V31CoreConfig,
    side: Side,
    trend_strength: int,
    entry_price: float,
    atr_1h: float,
    equity_for_risk: float,
) -> Tuple[float, float, float, int, float, float]:
    if atr_1h <= 0 or entry_price <= 0 or equity_for_risk <= 0:
        return 0.0, 0.0, 0.0, 0, 0.0, 0.0

    if trend_strength >= 2:
        rr = cfg.rr_strong
        sl_mult = cfg.sl_mult_strong
        pos_cap = cfg.pos_cap_strong
        max_bars = cfg.max_bars_strong
    else:
        rr = cfg.rr_normal
        sl_mult = cfg.sl_mult_normal
        pos_cap = cfg.pos_cap_normal
        max_bars = cfg.max_bars_normal

    sl_dist_abs = atr_1h * sl_mult
    sl_dist_pct_from_atr = sl_dist_abs / entry_price
    sl_dist_pct = max(sl_dist_pct_from_atr, cfg.min_sl_pct)
    sl_dist_abs = sl_dist_pct * entry_price

    if side == "LONG":
        stop_price = entry_price - sl_dist_abs
        take_price = entry_price + sl_dist_abs * rr
    else:
        stop_price = entry_price + sl_dist_abs
        take_price = entry_price - sl_dist_abs * rr

    if stop_price <= 0 or take_price <= 0:
        return 0.0, 0.0, 0.0, 0, 0.0, 0.0

    risk_target = equity_for_risk * cfg.risk_per_trade
    stop_dist_pct = abs(entry_price - stop_price) / entry_price
    if stop_dist_pct <= 0:
        return 0.0, 0.0, 0.0, 0, 0.0, 0.0

    notional_target = risk_target / stop_dist_pct
    max_notional = equity_for_risk * cfg.leverage * pos_cap
    notional = min(notional_target, max_notional)
    if notional <= 0:
        return 0.0, 0.0, 0.0, 0, 0.0, 0.0

    return stop_price, take_price, notional, max_bars, sl_dist_abs, rr
