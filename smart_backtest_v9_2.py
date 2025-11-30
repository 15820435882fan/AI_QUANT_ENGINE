#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SmartBacktest v9_2
===============================
第二季 · Step4.1（数据引擎升级版）
- 使用 real_market_data_v2.RealMarketData 进行多轮分页抓取 Binance K线
- 交易逻辑延续 v9_1：Trend + Structure 主导方向
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
# 0. 模拟 K 线（fallback）
# ============================================================
def generate_mock_data(symbol: str, days: int = 30, seed: Optional[int] = None):
    if seed is not None:
        np.random.seed(seed)
    n = days * 24 * 12
    prices = [100]
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
# 1. 结构：HH/HL/LH/LL + 假突破 + 震荡检测
# ============================================================
def compute_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    high = d["high"].values
    low = d["low"].values
    closes = d["close"].values
    opens = d["open"].values

    n = len(df)
    struct = ["none"] * n
    strength = np.zeros(n)
    fake_break = np.zeros(n)
    chop = np.zeros(n)

    for i in range(2, n):
        # 结构方向：三根K简单结构判定
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

        # 假突破：影线很长，实体较短
        body_high = max(opens[i], closes[i])
        body_low = min(opens[i], closes[i])
        wick_up = max(0.0, high[i] - body_high)
        wick_down = max(0.0, body_low - low[i])
        total_range = max(1e-9, high[i] - low[i])

        if (wick_up / total_range > 0.45) or (wick_down / total_range > 0.45):
            fake_break[i] = 1.0

        # 震荡区间：连续若干根实体都很小
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
# 2. 趋势指标 / ATR
# ============================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["ma_fast"] = d["close"].rolling(20).mean()
    d["ma_slow"] = d["close"].rolling(50).mean()
    d["trend_long_ok"] = d["ma_fast"] > d["ma_slow"]
    d["trend_short_ok"] = d["ma_fast"] < d["ma_slow"]

    hl = d["high"] - d["low"]
    hc = (d["high"] - d["close"].shift(1)).abs()
    lc = (d["low"] - d["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()

    ema20 = d["close"].ewm(span=20).mean()
    slope = (ema20 - ema20.shift(5)) / (ema20.shift(5).abs() + 1e-9)
    slope = (slope.abs() * 10).clip(0, 1)
    d["trend_strength"] = slope

    return d


# ============================================================
# 3. 策略信号
# ============================================================
def compute_strategy_signals(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close = d["close"]

    emaf = close.ewm(span=12).mean()
    emas = close.ewm(span=26).mean()
    macd = emaf - emas
    sig = macd.ewm(span=9).mean()
    hist = macd - sig
    d["sig_macd"] = np.where(hist > 0, 1, -1)

    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    d["sig_ema"] = np.where(ema20 > ema50, 1, -1)

    hh = d["high"].rolling(20).max()
    ll = d["low"].rolling(20).min()
    d["sig_turtle"] = np.where(
        close > hh.shift(1),
        1,
        np.where(close < ll.shift(1), -1, 0),
    )

    ma = close.rolling(20).mean()
    std = close.rolling(20).std()
    up = ma + 2 * std
    lo = ma - 2 * std
    d["sig_boll"] = np.where(
        close < lo,
        1,
        np.where(close > up, -1, 0),
    )

    rb_max = close.rolling(50).max()
    rb_min = close.rolling(50).min()
    d["sig_break"] = np.where(
        close > rb_max * 1.01,
        1,
        np.where(close < rb_min * 0.99, -1, 0),
    )

    return d


# ============================================================
# 4. ensemble（仅用于共识强度）
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
# 5. Alpha 评分：趋势 + 结构 主导
# ============================================================
def compute_alpha(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    ts = d["trend_strength"].values
    ss = d["structure_strength"].values
    cs = d["consensus_strength"].values
    atr = d["atr"].values
    close = d["close"].values
    fake = d["fake_break"].values
    chop = d["chop"].values
    struct = d["structure"].values

    n = len(d)
    a_long = np.zeros(n)
    a_short = np.zeros(n)

    for i in range(n):
        c = close[i]
        a = atr[i]
        vol = min(1.0, a / c * 200) if (c > 0 and not np.isnan(a)) else 0.0

        base_long = 0.45 * ts[i] + 0.35 * ss[i] + 0.15 * cs[i] + 0.05 * vol
        base_short = 0.45 * ts[i] + 0.35 * ss[i] + 0.15 * cs[i] + 0.05 * vol

        if struct[i] in ("HH", "HL"):
            base_long += 0.20
        if struct[i] in ("LL", "LH"):
            base_short += 0.20

        if fake[i] > 0.5:
            base_long *= 0.7
            base_short *= 0.7
        if chop[i] > 0.5:
            base_long *= 0.6
            base_short *= 0.6

        a_long[i] = min(max(base_long, 0.0), 1.2)
        a_short[i] = min(max(base_short, 0.0), 1.2)

    d["alpha_long"] = a_long
    d["alpha_short"] = a_short
    return d


# ============================================================
# 6. 交易引擎（沿用 v9_1）
# ============================================================
class AdaptiveSignalEngineV92:
    def __init__(self):
        self.base = {
            "sl_atr_mult": 1.8,
            "tp_atr_mult": 3.2,
            "trail_atr_mult": 1.8,
            "min_rr": 1.1,
            "alpha_threshold": 0.55,
            "base_risk": 0.01,
            "cooldown": 12 * 12,
            "max_loss_streak": 3,
        }
        self.override = {
            "ETH": {
                "base_risk": 0.006,
                "sl_atr_mult": 2.2,
                "tp_atr_mult": 4.2,
            }
        }

    def get_params(self, symbol: str) -> Dict[str, float]:
        P = self.base.copy()
        for k, v in self.override.items():
            if k in symbol.upper():
                P.update(v)
        return P

    def run(self, symbol: str, df: pd.DataFrame, capital: float) -> Dict[str, float]:
        P = self.get_params(symbol)

        d = compute_indicators(df)
        d = compute_market_structure(d)
        d = compute_strategy_signals(d)
        d = compute_ensemble(d)
        d = compute_alpha(d)

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

        for ts, row in d.iterrows():
            price = float(row["close"])
            atr = float(row["atr"]) if not np.isnan(row["atr"]) else 0.0

            # 持仓管理
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
                            cooldown = P["cooldown"]
                            loss_streak = 0
                            logger.info(
                                "🧊 %s 连续亏损 → 冷静期 %d bars",
                                symbol,
                                cooldown,
                            )

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

            # 空仓：考虑开仓
            if pos == 0:
                if cooldown > 0:
                    cooldown -= 1
                    continue
                if atr <= 0:
                    continue

                aL = float(row["alpha_long"])
                aS = float(row["alpha_short"])
                aL = max(0.0, min(aL, 1.2))
                aS = max(0.0, min(aS, 1.2))
                best = max(aL, aS)
                if best < P["alpha_threshold"]:
                    continue

                long_sig = aL >= aS

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

                risk_amt = cash * P["base_risk"]
                size = risk_amt / sl_dist
                if size <= 0:
                    continue

                pos = 1 if long_sig else -1
                entry = price
                sl = sl_c
                tp = tp_c

        return {
            "pnl": pnl_total,
            "trades": trades,
            "wins": wins,
            "win_rate": wins / trades * 100 if trades > 0 else 0.0,
            "max_dd": max_dd,
        }


# ============================================================
# 7. 多币种回测
# ============================================================
def run_backtest(symbols, days, capital, seed, source):
    logger.info("🚀 SmartBacktest V9_2 启动")
    logger.info("🪙 币种: %s", symbols)

    market = RealMarketData()
    engine = AdaptiveSignalEngineV92()

    results = {}
    each_cap = capital / len(symbols)

    for sym in symbols:
        logger.info("🔍 处理 %s", sym)
        try:
            if source == "real":
                df = market.get_recent_klines(sym, "5m", days)
                print(f"📥 最终获取K线: {sym}, {len(df)} 行")
            else:
                df = generate_mock_data(sym, days, seed)
        except Exception as e:
            logger.error("❌ 获取 %s 数据失败: %s，使用模拟数据", sym, e)
            df = generate_mock_data(sym, days, seed)

        res = engine.run(sym, df, each_cap)
        results[sym] = res

    total_pnl = sum(r["pnl"] for r in results.values())
    total_trades = sum(r["trades"] for r in results.values())
    total_wins = sum(r["wins"] for r in results.values())
    win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0.0
    max_dd = min(r["max_dd"] for r in results.values()) if results else 0.0

    print("\n========== 📈 SmartBacktest V9_2 报告 ==========")
    print(f"总收益: {total_pnl:.2f}")
    print(f"总交易数: {total_trades}")
    print(f"总胜率: {win_rate:.2f}%")
    print(f"最大回撤: {max_dd:.2f}%")

    print("\n按币种：")
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
    pa = argparse.ArgumentParser(description="SmartBacktest V9_2")
    pa.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    pa.add_argument("--days", type=int, default=30)
    pa.add_argument("--initial-capital", type=float, default=10000)
    pa.add_argument("--seed", type=int, default=None)
    pa.add_argument("--data-source", type=str, choices=["real", "mock"], default="real")
    a = pa.parse_args()

    run_backtest(
        parse_symbols(a.symbols),
        a.days,
        a.initial_capital,
        a.seed,
        a.data_source,
    )


if __name__ == "__main__":
    main()
