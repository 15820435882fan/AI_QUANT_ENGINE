#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SmartBacktest v11
====================================
Multi-Timeframe AI Trend Engine

- 使用 4h / 1h / 5m 三周期
- 4h 负责大趋势方向过滤（只能顺 4h 做单）
- 1h 负责中周期结构确认（同向加权，反向削弱）
- 5m 负责精细入场（结构 + 波动 + 因子合成）

在 v10 的基础上：
- 保留 Anti-Noise + 冷静期 + 交易频率门控
- 引入 MTF 上下文：HTF / MTF → LTF 映射
"""

import os
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

import argparse
import logging
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from real_market_data_v2 import RealMarketData

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# 0. fallback 模拟 K 线
# ============================================================
def generate_mock_data(symbol: str, days: int = 30, seed: Optional[int] = None):
    if seed is not None:
        np.random.seed(seed)
    n = days * 24 * 12  # 5m bars
    prices = [100.0]
    for _ in range(n):
        prices.append(prices[-1] * (1 + np.random.normal(0, 1) * 0.001))

    prices = np.array(prices)
    df = pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=n, freq="5min"),
        "open": prices[:-1],
        "high": np.maximum(prices[:-1], prices[1:]),
        "low": np.minimum(prices[:-1], prices[1:]),
        "close": prices[1:],
        "volume": np.random.rand(n),
    })
    return df.set_index("timestamp")


# ============================================================
# 1. 通用指标：趋势 / ATR
# ============================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ma_fast"] = d["close"].rolling(20).mean()
    d["ma_slow"] = d["close"].rolling(50).mean()
    d["trend_long_ok"] = d["ma_fast"] > d["ma_slow"]
    d["trend_short_ok"] = d["ma_fast"] < d["ma_slow"]

    # ATR
    hl = d["high"] - d["low"]
    hc = (d["high"] - d["close"].shift(1)).abs()
    lc = (d["low"] - d["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()

    # 趋势强度（EMA20 斜率）
    ema20 = d["close"].ewm(span=20).mean()
    slope = (ema20 - ema20.shift(5)) / (ema20.shift(5).abs() + 1e-9)
    slope = (slope.abs() * 10).clip(0, 1)
    d["trend_strength"] = slope

    return d


# ============================================================
# 2. 价格结构：HH/HL/LH/LL + 假突破 + 震荡
# ============================================================
def compute_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    high = d["high"].values
    low = d["low"].values
    closes = d["close"].values
    opens = d["open"].values

    n = len(d)
    struct = ["none"] * n
    strength = np.zeros(n)
    fake_break = np.zeros(n)
    chop = np.zeros(n)

    for i in range(2, n):
        # 结构方向
        if high[i] > high[i - 1] and high[i - 1] > high[i - 2]:
            struct[i] = "HH"
            strength[i] = 1.0
        elif low[i] > low[i - 1] and low[i - 1] > low[i - 2]:
            struct[i] = "HL"
            strength[i] = 0.7
        elif high[i] < high[i - 1] and high[i - 1] < high[i - 2]:
            struct[i] = "LH"
            strength[i] = 0.7
        elif low[i] < low[i - 1] and low[i - 1] < low[i - 2]:
            struct[i] = "LL"
            strength[i] = 1.0

        # 假突破：长影线 + 短实体
        body_high = max(opens[i], closes[i])
        body_low = min(opens[i], closes[i])
        wick_up = max(0.0, high[i] - body_high)
        wick_down = max(0.0, body_low - low[i])
        total_range = max(1e-9, high[i] - low[i])

        if (wick_up / total_range > 0.45) or (wick_down / total_range > 0.45):
            fake_break[i] = 1.0

        # 震荡区：连续多根实体很小
        if i > 5:
            bodies = np.abs(d["close"].iloc[i-5:i] - d["open"].iloc[i-5:i])
            rng = (d["high"].iloc[i-5:i] - d["low"].iloc[i-5:i]).mean()
            if rng > 0 and bodies.mean() / rng < 0.25:
                chop[i] = 1.0

    d["structure"] = struct
    d["structure_strength"] = strength
    d["fake_break"] = fake_break
    d["chop"] = chop
    return d


# ============================================================
# 3. 策略信号（MACD / EMA / Turtle / Boll / Breakout）
# ============================================================
def compute_strategy_signals(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close = d["close"]

    # MACD
    emaf = close.ewm(span=12).mean()
    emas = close.ewm(span=26).mean()
    macd = emaf - emas
    sig = macd.ewm(span=9).mean()
    hist = macd - sig
    d["sig_macd"] = np.where(hist > 0, 1, -1)

    # EMA Cross
    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    d["sig_ema"] = np.where(ema20 > ema50, 1, -1)

    # Turtle
    hh = d["high"].rolling(20).max()
    ll = d["low"].rolling(20).min()
    d["sig_turtle"] = np.where(
        close > hh.shift(1),
        1,
        np.where(close < ll.shift(1), -1, 0),
    )

    # Boll
    ma = close.rolling(20).mean()
    std = close.rolling(20).std()
    up = ma + 2 * std
    lo = ma - 2 * std
    d["sig_boll"] = np.where(
        close < lo,
        1,
        np.where(close > up, -1, 0),
    )

    # Breakout
    rb_max = close.rolling(50).max()
    rb_min = close.rolling(50).min()
    d["sig_break"] = np.where(
        close > rb_max * 1.01,
        1,
        np.where(close < rb_min * 0.99, -1, 0),
    )

    return d


# ============================================================
# 4. ensemble（只做共识强度）
# ============================================================
def compute_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    sigs = ["sig_macd", "sig_ema", "sig_turtle", "sig_boll", "sig_break"]
    arr = np.zeros(len(d))
    for s in sigs:
        arr += d[s].fillna(0).values
    raw = arr / len(sigs)
    d["ensemble_raw"] = raw
    d["consensus_strength"] = np.abs(raw).clip(0, 1)
    return d


# ============================================================
# 5. HTF / MTF：趋势方向 + Regime 标注
# ============================================================
def compute_tf_trend_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    给一个任意周期的 df（已算好 indicators + structure），
    输出：
    - tf_trend_dir: +1/-1/0
    - tf_regime: up_trend / down_trend / range / choppy
    """
    d = df.copy()
    trend_dir = np.zeros(len(d), dtype=int)
    regime = np.array(["unknown"] * len(d), dtype=object)

    for i, row in enumerate(d.itertuples()):
        ts = getattr(row, "trend_strength")
        chop = getattr(row, "chop")
        s = getattr(row, "structure")
        ma_fast = getattr(row, "ma_fast")
        ma_slow = getattr(row, "ma_slow")

        # 方向判定
        dir_ = 0
        if not np.isnan(ma_fast) and not np.isnan(ma_slow):
            if ma_fast > ma_slow and s in ("HH", "HL"):
                dir_ = 1
            elif ma_fast < ma_slow and s in ("LL", "LH"):
                dir_ = -1
            else:
                dir_ = 0
        trend_dir[i] = dir_

        # Regime
        if chop > 0.5:
            regime[i] = "choppy"
        else:
            if ts >= 0.6:
                regime[i] = "strong_trend_up" if dir_ > 0 else (
                    "strong_trend_down" if dir_ < 0 else "strong_trend_flat"
                )
            elif ts >= 0.3:
                regime[i] = "weak_trend_up" if dir_ > 0 else (
                    "weak_trend_down" if dir_ < 0 else "weak_trend_flat"
                )
            else:
                regime[i] = "range"

    d["tf_trend_dir"] = trend_dir
    d["tf_regime"] = regime
    return d


# ============================================================
# 6. LTF：Regime + Alpha（但会结合 HTF / MTF 做修正）
# ============================================================
def compute_ltf_regime_alpha(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 5m 级别计算：
    - 基础 regime（choppy / low_vol / high_vol / trend / range）
    - 基础 alpha_long / alpha_short （不含 HTF 信息）
    然后会在交易引擎里再结合 HTF / MTF 做修正。
    """
    d = df.copy()

    close = d["close"].values
    atr = d["atr"].values
    ts = d["trend_strength"].values
    ss = d["structure_strength"].values
    cs = d["consensus_strength"].values
    fake = d["fake_break"].values
    chop = d["chop"].values

    n = len(d)
    regime = np.array(["unknown"] * n, dtype=object)
    vol_level = np.zeros(n)
    alpha_long = np.zeros(n)
    alpha_short = np.zeros(n)

    atr_series = pd.Series(atr)
    atr_mean = atr_series.rolling(100, min_periods=20).mean().values

    for i in range(n):
        c = close[i]
        a = atr[i]
        if c > 0 and not np.isnan(a):
            v = a / c * 200.0
        else:
            v = 0.0
        v = max(0.0, min(v, 2.5))
        vol_level[i] = v

        if atr_mean[i] > 0:
            vol_ratio = a / atr_mean[i]
        else:
            vol_ratio = 1.0

        # Regime
        if chop[i] > 0.5:
            reg = "choppy"
        elif vol_ratio < 0.5:
            reg = "low_vol"
        elif vol_ratio > 1.8:
            reg = "high_vol"
        elif ts[i] > 0.6:
            reg = "strong_trend"
        elif ts[i] > 0.3:
            reg = "weak_trend"
        else:
            reg = "range"

        regime[i] = reg

        # 基础 alpha
        base_long = (
            0.45 * ts[i]
            + 0.35 * ss[i]
            + 0.15 * cs[i]
            + 0.05 * (1.0 - abs(v - 0.8))
        )
        base_short = (
            0.45 * ts[i]
            + 0.35 * ss[i]
            + 0.15 * cs[i]
            + 0.05 * (1.0 - abs(v - 0.8))
        )

        if d["structure"].iloc[i] in ("HH", "HL"):
            base_long += 0.2
        if d["structure"].iloc[i] in ("LL", "LH"):
            base_short += 0.2

        if fake[i] > 0.5:
            base_long *= 0.7
            base_short *= 0.7

        if reg == "choppy":
            base_long *= 0.5
            base_short *= 0.5
        if reg == "low_vol":
            base_long *= 0.4
            base_short *= 0.4

        alpha_long[i] = min(max(base_long, 0.0), 1.5)
        alpha_short[i] = min(max(base_short, 0.0), 1.5)

    d["ltf_regime"] = regime
    d["ltf_vol_level"] = vol_level
    d["alpha_long_base"] = alpha_long
    d["alpha_short_base"] = alpha_short
    return d


# ============================================================
# 7. MTF 上下文对齐：将 4h / 1h 映射到 5m 时间轴
# ============================================================
def align_tf_to_ltf(ltf: pd.DataFrame, tf: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    把高周期 tf（4h / 1h）的关键信息映射到 5m 时间轴：
    - 使用 reindex(method='ffill') 对齐
    """
    cols = ["tf_trend_dir", "tf_regime", "trend_strength"]
    sub = tf[cols].copy()
    sub.columns = [f"{prefix}_{c}" for c in cols]
    aligned = sub.reindex(ltf.index, method="ffill")
    return aligned


# ============================================================
# 8. 交易引擎：MTF + Anti-Noise
# ============================================================
class AdaptiveSignalEngineV11:
    def __init__(self):
        self.base = {
            "sl_atr_mult": 1.8,
            "tp_atr_mult": 3.2,
            "trail_atr_mult": 1.8,
            "min_rr": 1.2,
            "alpha_threshold": 0.65,
            "base_risk": 0.01,
            "cooldown_bars": 12 * 12,
            "max_loss_streak": 3,
            "min_bars_between_trades": 15,
        }
        self.override = {
            "BTC": {
                "base_risk": 0.005,
                "alpha_threshold": 0.75,
                "sl_atr_mult": 2.6,
                "tp_atr_mult": 5.2,
                "min_bars_between_trades": 20,
            },
            "ETH": {
                "base_risk": 0.006,
                "alpha_threshold": 0.62,
                "sl_atr_mult": 2.2,
                "tp_atr_mult": 4.4,
                "min_bars_between_trades": 12,
            },
        }

    def get_params(self, symbol: str) -> Dict[str, float]:
        P = self.base.copy()
        usym = symbol.upper()
        for k, v in self.override.items():
            if k in usym:
                P.update(v)
        return P

    def run(self, symbol: str, df_ltf: pd.DataFrame, df_mtf: pd.DataFrame,
            df_htf: pd.DataFrame, capital: float) -> Dict[str, float]:
        P = self.get_params(symbol)

        if df_ltf is None or len(df_ltf) == 0:
            logger.warning("%s LTF 数据为空，跳过回测", symbol)
            return {"pnl": 0.0, "trades": 0, "wins": 0, "win_rate": 0.0, "max_dd": 0.0}

        # 1) 先对 HTF / MTF 做 trend + regime
        htf = compute_indicators(df_htf)
        htf = compute_market_structure(htf)
        htf = compute_tf_trend_regime(htf)

        mtf = compute_indicators(df_mtf)
        mtf = compute_market_structure(mtf)
        mtf = compute_tf_trend_regime(mtf)

        # 2) LTF 计算所有本地因子
        ltf = compute_indicators(df_ltf)
        ltf = compute_market_structure(ltf)
        ltf = compute_strategy_signals(ltf)
        ltf = compute_ensemble(ltf)
        ltf = compute_ltf_regime_alpha(ltf)

        # 3) 将 HTF / MTF 映射到 LTF 时间轴
        h_ctx = align_tf_to_ltf(ltf, htf, "htf")
        m_ctx = align_tf_to_ltf(ltf, mtf, "mtf")
        ltf = pd.concat([ltf, h_ctx, m_ctx], axis=1)

        # 4) 开始交易循环
        cash = capital
        pos = 0
        size = 0.0
        entry = 0.0
        sl = 0.0
        tp = 0.0

        pnl_total = 0.0
        trades = 0
        wins = 0
        loss_streak = 0
        cooldown = 0

        eq = capital
        max_eq = capital
        max_dd = 0.0
        last_entry_bar: Optional[int] = None

        for bar_index, (ts, row) in enumerate(ltf.itertuples()):
            price = float(getattr(row, "close"))
            atr = float(getattr(row, "atr")) if not np.isnan(getattr(row, "atr")) else 0.0

            ltf_regime = getattr(row, "ltf_regime")
            fake_break = float(getattr(row, "fake_break"))
            consensus = float(getattr(row, "consensus_strength"))

            alpha_long_base = float(getattr(row, "alpha_long_base"))
            alpha_short_base = float(getattr(row, "alpha_short_base"))

            htf_dir = int(getattr(row, "htf_tf_trend_dir")) if not np.isnan(getattr(row, "htf_tf_trend_dir")) else 0
            htf_regime = getattr(row, "htf_tf_regime")

            mtf_dir = int(getattr(row, "mtf_tf_trend_dir")) if not np.isnan(getattr(row, "mtf_tf_trend_dir")) else 0
            mtf_regime = getattr(row, "mtf_tf_regime")

            # ------------------ 持仓管理 ------------------
            if pos != 0:
                if atr > 0 and P["trail_atr_mult"] > 0:
                    if pos > 0:
                        sl = max(sl, price - P["trail_atr_mult"] * atr)
                    else:
                        sl = min(sl, price + P["trail_atr_mult"] * atr)

                exit_flag = False
                if pos > 0 and (price <= sl or price >= tp):
                    exit_flag = True
                if pos < 0 and (price >= sl or price <= tp):
                    exit_flag = True

                if exit_flag:
                    pnl = (price - entry) * size * pos
                    pnl_total += pnl
                    cash += pnl
                    trades += 1

                    if pnl > 0:
                        wins += 1
                        loss_streak = 0
                    else:
                        loss_streak += 1
                        if loss_streak >= P["max_loss_streak"]:
                            cooldown = P["cooldown_bars"]
                            loss_streak = 0
                            logger.info("🧊 %s 连续亏损 → 冷静期 %d bars", symbol, cooldown)

                    pos = 0
                    size = 0.0
                    entry = 0.0
                    sl = 0.0
                    tp = 0.0

            # 更新权益 / 回撤
            if pos != 0:
                eq = cash + (price - entry) * size * pos
            else:
                eq = cash
            max_eq = max(max_eq, eq)
            if max_eq > 0:
                dd = (eq - max_eq) / max_eq * 100.0
                max_dd = min(max_dd, dd)

            # ------------------ 空仓：考虑是否开新仓 ------------------
            if pos == 0:
                if cooldown > 0:
                    cooldown -= 1
                    continue
                if atr <= 0:
                    continue

                # 交易频率门控
                if last_entry_bar is not None:
                    if bar_index - last_entry_bar < P["min_bars_between_trades"]:
                        continue

                # ① 高周期方向过滤：必须有 4h 方向，且不是 choppy
                if htf_dir == 0:
                    continue
                if isinstance(htf_regime, str) and ("choppy" in htf_regime):
                    continue

                # ② 本地 regime 降噪：choppy / low_vol 直接不做
                if ltf_regime in ("choppy", "low_vol"):
                    continue

                # ③ 假突破 + 共识弱：强制过滤
                if fake_break > 0.5 and consensus < 0.6:
                    continue

                # ④ 基础 alpha 修正：加入 HTF / MTF 权重
                alpha_long = alpha_long_base
                alpha_short = alpha_short_base

                # HTF 大方向：只允许顺势方向
                if htf_dir > 0:
                    alpha_short *= 0.2
                    alpha_long *= 1.1
                elif htf_dir < 0:
                    alpha_long *= 0.2
                    alpha_short *= 1.1

                # MTF 中周期：同向加权，反向削弱
                if htf_dir != 0 and mtf_dir != 0:
                    if htf_dir == mtf_dir:
                        alpha_long *= 1.1
                        alpha_short *= 1.1
                    else:
                        alpha_long *= 0.6
                        alpha_short *= 0.6

                # MTF Regime：弱趋势 / strong_trend 稍微放宽
                dyn_alpha_th = P["alpha_threshold"]
                if isinstance(mtf_regime, str):
                    if "strong_trend" in mtf_regime:
                        dyn_alpha_th *= 0.9
                    elif "weak_trend" in mtf_regime:
                        dyn_alpha_th *= 0.95

                # 最终 alpha 比较
                alpha_long = max(0.0, min(alpha_long, 2.0))
                alpha_short = max(0.0, min(alpha_short, 2.0))
                best_alpha = max(alpha_long, alpha_short)

                if best_alpha < dyn_alpha_th:
                    continue

                # 最终方向必须与 HTF 一致（顺势）
                long_sig = alpha_long >= alpha_short
                if htf_dir > 0 and not long_sig:
                    continue
                if htf_dir < 0 and long_sig:
                    continue

                # 止损 / 止盈
                if long_sig:
                    sl_c = price - P["sl_atr_mult"] * atr
                    tp_c = price + P["tp_atr_mult"] * atr
                else:
                    sl_c = price + P["sl_atr_mult"] * atr
                    tp_c = price - P["tp_atr_mult"] * atr

                sl_dist = abs(price - sl_c)
                tp_dist = abs(tp_c - price)
                if sl_dist <= 0 or tp_dist <= 0:
                    continue

                rr = tp_dist / sl_dist
                if rr < P["min_rr"]:
                    continue

                # 风险动态：trend 优先，高波动适当降风险
                dyn_base_risk = P["base_risk"]
                if isinstance(htf_regime, str) and "strong_trend" in htf_regime:
                    dyn_base_risk *= 1.3
                elif isinstance(htf_regime, str) and "weak_trend" in htf_regime:
                    dyn_base_risk *= 1.1

                risk_amt = cash * dyn_base_risk
                if risk_amt <= 0:
                    continue

                size = risk_amt / sl_dist
                if size <= 0:
                    continue

                pos = 1 if long_sig else -1
                entry = price
                sl = sl_c
                tp = tp_c
                last_entry_bar = bar_index

        return {
            "pnl": pnl_total,
            "trades": trades,
            "wins": wins,
            "win_rate": wins / trades * 100 if trades > 0 else 0.0,
            "max_dd": max_dd,
        }


# ============================================================
# 9. 多周期数据加载 & 回测入口
# ============================================================
def run_backtest(symbols: List[str], days: int, capital: float,
                 seed: Optional[int], source: str):
    logger.info("🚀 SmartBacktest V11 启动")
    logger.info("🪙 币种: %s", symbols)
    logger.info("📅 回测天数: %d", days)
    logger.info("📊 数据源: %s", source)

    market = RealMarketData()
    engine = AdaptiveSignalEngineV11()

    results: Dict[str, Dict[str, float]] = {}
    each_cap = capital / len(symbols) if symbols else capital

    for sym in symbols:
        logger.info("🔍 处理 %s", sym)
        try:
            if source == "real":
                # 5m 作为 LTF
                df_ltf = market.get_recent_klines(sym, "5m", days)
                # 1h 作为 MTF
                df_mtf = market.get_recent_klines(sym, "1h", days + 3)
                # 4h 作为 HTF（多拉几天缓冲）
                df_htf = market.get_recent_klines(sym, "4h", days + 7)

                print(f"📥 {sym} 5m={len(df_ltf)}, 1h={len(df_mtf)}, 4h={len(df_htf)}")
            else:
                df_ltf = generate_mock_data(sym, days, seed)
                df_mtf = df_ltf.resample("1H").agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }).dropna()
                df_htf = df_ltf.resample("4H").agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }).dropna()
        except Exception as e:
            logger.error("❌ 获取 %s 数据失败: %s，使用模拟数据", sym, e)
            df_ltf = generate_mock_data(sym, days, seed)
            df_mtf = df_ltf.resample("1H").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()
            df_htf = df_ltf.resample("4H").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()

        res = engine.run(sym, df_ltf, df_mtf, df_htf, each_cap)
        results[sym] = res

    total_pnl = sum(r["pnl"] for r in results.values())
    total_trades = sum(r["trades"] for r in results.values())
    total_wins = sum(r["wins"] for r in results.values())
    win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0.0
    max_dd = min(r["max_dd"] for r in results.values()) if results else 0.0

    print("\n========== 📈 SmartBacktest V11 报告 ==========")
    print(f"总收益: {total_pnl:.2f}")
    print(f"总交易数: {total_trades}")
    print(f"总胜率: {win_rate:.2f}%")
    print(f"最大回撤: {max_dd:.2f}%\n")

    print("按币种：")
    for sym, r in results.items():
        print(
            f"- {sym}: pnl={r['pnl']:.2f}, trades={r['trades']}, "
            f"win={r['win_rate']:.2f}%, DD={r['max_dd']:.2f}%"
        )

    return results


# ============================================================
# MAIN
# ============================================================
def parse_symbols(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    pa = argparse.ArgumentParser(description="SmartBacktest V11")
    pa.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    pa.add_argument("--days", type=int, default=30)
    pa.add_argument("--initial-capital", type=float, default=10000)
    pa.add_argument("--seed", type=int, default=None)
    pa.add_argument("--data-source", type=str, choices=["real", "mock"], default="real")
    args = pa.parse_args()

    run_backtest(
        symbols=parse_symbols(args.symbols),
        days=args.days,
        capital=args.initial_capital,
        seed=args.seed,
        source=args.data_source,
    )


if __name__ == "__main__":
    main()
