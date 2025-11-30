#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SmartBacktest v11_1
Multi-Timeframe Trend Engine (4h + 1h + 5m)
- 修复 itertuples unpack bug (ValueError)
- 改用 row.<col> 方式取值
- 性能优化（减少列访问次数）
"""

import os
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

import argparse
import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict

from real_market_data_v2 import RealMarketData

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


############################################################
# 0. fallback mock
############################################################
def generate_mock_data(symbol: str, days: int = 30, seed: Optional[int] = None):
    if seed:
        np.random.seed(seed)
    n = days * 24 * 12
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


############################################################
# 1. Indicators
############################################################
def compute_indicators(df):
    d = df.copy()

    # MA
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

    # EMA slope
    ema20 = d["close"].ewm(span=20).mean()
    slope = (ema20 - ema20.shift(5)) / (ema20.shift(5).abs() + 1e-9)
    slope = (slope.abs() * 10).clip(0, 1)
    d["trend_strength"] = slope

    return d


############################################################
# 2. Market Structure（HH/HL/LH/LL）
############################################################
def compute_market_structure(df):
    d = df.copy()
    high = d["high"].values
    low = d["low"].values
    opens = d["open"].values
    closes = d["close"].values

    n = len(d)
    struct = ["none"] * n
    ss = np.zeros(n)
    fake = np.zeros(n)
    chop = np.zeros(n)

    for i in range(2, n):
        # structure
        if high[i] > high[i - 1] and high[i - 1] > high[i - 2]:
            struct[i] = "HH"
            ss[i] = 1.0
        elif low[i] > low[i - 1] and low[i - 1] > low[i - 2]:
            struct[i] = "HL"
            ss[i] = 0.7
        elif high[i] < high[i - 1] and high[i - 1] < high[i - 2]:
            struct[i] = "LH"
            ss[i] = 0.7
        elif low[i] < low[i - 1] and low[i - 1] < low[i - 2]:
            struct[i] = "LL"
            ss[i] = 1.0

        # fake break
        body_high = max(opens[i], closes[i])
        body_low = min(opens[i], closes[i])
        wick_up = max(0.0, high[i] - body_high)
        wick_down = max(0.0, body_low - low[i])
        total = max(high[i] - low[i], 1e-9)

        if wick_up / total > 0.45 or wick_down / total > 0.45:
            fake[i] = 1.0

        # choppy
        if i > 5:
            bodies = np.abs(d["close"].iloc[i - 5:i] - d["open"].iloc[i - 5:i])
            rng = (d["high"].iloc[i - 5:i] - d["low"].iloc[i - 5:i]).mean()
            if rng > 0 and bodies.mean() / rng < 0.25:
                chop[i] = 1.0

    d["structure"] = struct
    d["structure_strength"] = ss
    d["fake_break"] = fake
    d["chop"] = chop
    return d


############################################################
# 3. Strategy Signals
############################################################
def compute_strategy_signals(df):
    d = df.copy()

    close = d["close"]

    # MACD
    emaf = close.ewm(span=12).mean()
    emas = close.ewm(span=26).mean()
    macd = emaf - emas
    sig = macd.ewm(span=9).mean()
    hist = macd - sig
    d["sig_macd"] = np.where(hist > 0, 1, -1)

    # EMA cross
    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    d["sig_ema"] = np.where(ema20 > ema50, 1, -1)

    # Turtle
    hh = d["high"].rolling(20).max()
    ll = d["low"].rolling(20).min()
    d["sig_turtle"] = np.where(
        close > hh.shift(1),
        1,
        np.where(close < ll.shift(1), -1, 0)
    )

    # Boll
    ma = close.rolling(20).mean()
    std = close.rolling(20).std()
    up = ma + 2 * std
    lo = ma - 2 * std
    d["sig_boll"] = np.where(
        close < lo,
        1,
        np.where(close > up, -1, 0)
    )

    # Breakout
    roll_max = close.rolling(50).max()
    roll_min = close.rolling(50).min()
    d["sig_break"] = np.where(
        close > roll_max * 1.01,
        1,
        np.where(close < roll_min * 0.99, -1, 0)
    )

    return d


############################################################
# 4. ensemble
############################################################
def compute_ensemble(df):
    d = df.copy()
    sigs = ["sig_macd", "sig_ema", "sig_turtle", "sig_boll", "sig_break"]
    arr = np.zeros(len(d))
    for s in sigs:
        arr += d[s].fillna(0).values
    raw = arr / len(sigs)
    d["ensemble_raw"] = raw
    d["consensus_strength"] = np.abs(raw).clip(0, 1)
    return d


############################################################
# 5. Trend/Regime for HTF & MTF
############################################################
def compute_tf_trend_regime(df):
    d = df.copy()
    dir_ = np.zeros(len(d), dtype=int)
    regime = np.array(["unknown"] * len(d), dtype=object)

    for i, row in enumerate(d.itertuples()):
        ts = row.trend_strength
        chop = row.chop
        s = row.structure
        ma_fast = row.ma_fast
        ma_slow = row.ma_slow

        # Trend direction
        tdir = 0
        if not np.isnan(ma_fast) and not np.isnan(ma_slow):
            if ma_fast > ma_slow and s in ("HH", "HL"):
                tdir = 1
            elif ma_fast < ma_slow and s in ("LL", "LH"):
                tdir = -1
        dir_[i] = tdir

        # Regime
        if chop > 0.5:
            regime[i] = "choppy"
        else:
            if ts >= 0.6:
                regime[i] = "strong_trend_up" if tdir > 0 else (
                    "strong_trend_down" if tdir < 0 else "strong_trend_flat"
                )
            elif ts >= 0.3:
                regime[i] = "weak_trend_up" if tdir > 0 else (
                    "weak_trend_down" if tdir < 0 else "weak_trend_flat"
                )
            else:
                regime[i] = "range"

    d["tf_trend_dir"] = dir_
    d["tf_regime"] = regime
    return d


############################################################
# 6. LTF Alpha/Regime
############################################################
def compute_ltf_regime_alpha(df):
    d = df.copy()

    close = d.close.values
    atr = d.atr.values
    ts = d.trend_strength.values
    ss = d.structure_strength.values
    cs = d.consensus_strength.values
    fake = d.fake_break.values
    chop = d.chop.values

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

        # volatility
        if c > 0 and not np.isnan(a):
            v = a / c * 200.0
        else:
            v = 0
        v = max(0.0, min(v, 2.5))
        vol_level[i] = v

        # atr ratio
        if atr_mean[i] > 0:
            vol_ratio = a / atr_mean[i]
        else:
            vol_ratio = 1.0

        # regime
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

        # base alpha
        al = (
            0.45 * ts[i] +
            0.35 * ss[i] +
            0.15 * cs[i] +
            0.05 * (1 - abs(v - 0.8))
        )
        as_ = al

        if d.structure.iloc[i] in ("HH", "HL"):
            al += 0.2
        if d.structure.iloc[i] in ("LL", "LH"):
            as_ += 0.2

        if fake[i] > 0.5:
            al *= 0.7
            as_ *= 0.7

        if reg == "choppy":
            al *= 0.5
            as_ *= 0.5
        if reg == "low_vol":
            al *= 0.4
            as_ *= 0.4

        alpha_long[i] = min(max(al, 0), 1.5)
        alpha_short[i] = min(max(as_, 0), 1.5)

    d["ltf_regime"] = regime
    d["ltf_vol_level"] = vol_level
    d["alpha_long_base"] = alpha_long
    d["alpha_short_base"] = alpha_short
    return d


############################################################
# 7. TF Alignment
############################################################
def align_tf_to_ltf(ltf, tf, prefix):
    cols = ["tf_trend_dir", "tf_regime", "trend_strength"]
    sub = tf[cols].copy()
    sub.columns = [f"{prefix}_{c}" for c in cols]
    return sub.reindex(ltf.index, method="ffill")


############################################################
# 8. Trading Engine
############################################################
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

    def get_params(self, symbol: str):
        P = self.base.copy()
        usym = symbol.upper()
        for k, v in self.override.items():
            if k in usym:
                P.update(v)
        return P

    def run(self, symbol, df_ltf, df_mtf, df_htf, capital):
        #######################################
        # Build MTF layers
        #######################################
        if df_ltf is None or len(df_ltf) == 0:
            return {"pnl": 0, "trades": 0, "wins": 0, "win_rate": 0, "max_dd": 0}

        P = self.get_params(symbol)

        # HTF
        htf = compute_indicators(df_htf)
        htf = compute_market_structure(htf)
        htf = compute_tf_trend_regime(htf)

        # MTF
        mtf = compute_indicators(df_mtf)
        mtf = compute_market_structure(mtf)
        mtf = compute_tf_trend_regime(mtf)

        # LTF
        ltf = compute_indicators(df_ltf)
        ltf = compute_market_structure(ltf)
        ltf = compute_strategy_signals(ltf)
        ltf = compute_ensemble(ltf)
        ltf = compute_ltf_regime_alpha(ltf)

        # TF 映射
        h_ctx = align_tf_to_ltf(ltf, htf, "htf")
        m_ctx = align_tf_to_ltf(ltf, mtf, "mtf")
        ltf = pd.concat([ltf, h_ctx, m_ctx], axis=1)

        #######################################
        # Trading Loop (修复 row 解包)
        #######################################
        cash = capital
        pos = 0
        size = 0
        entry = 0
        sl = 0
        tp = 0

        pnl_total = 0
        trades = 0
        wins = 0
        loss_streak = 0
        cooldown = 0

        eq = capital
        max_eq = capital
        max_dd = 0
        last_entry_bar = None

        for bar_index, row in enumerate(ltf.itertuples()):
            price = row.close
            atr = row.atr if not np.isnan(row.atr) else 0

            # Local vars
            ltf_regime = row.ltf_regime
            fake = row.fake_break
            consensus = row.consensus_strength
            alpha_long_base = row.alpha_long_base
            alpha_short_base = row.alpha_short_base

            htf_dir = row.htf_tf_trend_dir if not np.isnan(row.htf_tf_trend_dir) else 0
            htf_regime = row.htf_tf_regime

            mtf_dir = row.mtf_tf_trend_dir if not np.isnan(row.mtf_tf_trend_dir) else 0
            mtf_regime = row.mtf_tf_regime

            #######################################
            # Manage position
            #######################################
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
                    size = 0
                    entry = 0
                    sl = 0
                    tp = 0

            #######################################
            # Update DD
            #######################################
            if pos != 0:
                eq = cash + (price - entry) * size * pos
            else:
                eq = cash
            max_eq = max(max_eq, eq)
            if max_eq > 0:
                dd = (eq - max_eq) / max_eq * 100
                max_dd = min(max_dd, dd)

            #######################################
            # Entry logic
            #######################################
            if pos == 0:
                if cooldown > 0:
                    cooldown -= 1
                    continue
                if atr <= 0:
                    continue

                # Trading frequency gate
                if last_entry_bar is not None:
                    if bar_index - last_entry_bar < P["min_bars_between_trades"]:
                        continue

                # High timeframe direction must exist
                if htf_dir == 0:
                    continue
                if isinstance(htf_regime, str) and ("choppy" in htf_regime):
                    continue

                # LTF filter
                if ltf_regime in ("choppy", "low_vol"):
                    continue

                if fake > 0.5 and consensus < 0.6:
                    continue

                # Base alpha
                al = alpha_long_base
                as_ = alpha_short_base

                # HTF Direction
                if htf_dir > 0:
                    as_ *= 0.2
                    al *= 1.1
                elif htf_dir < 0:
                    al *= 0.2
                    as_ *= 1.1

                # MTF Direction
                if htf_dir != 0 and mtf_dir != 0:
                    if htf_dir == mtf_dir:
                        al *= 1.1
                        as_ *= 1.1
                    else:
                        al *= 0.6
                        as_ *= 0.6

                dyn_alpha_th = P["alpha_threshold"]
                if isinstance(mtf_regime, str):
                    if "strong_trend" in mtf_regime:
                        dyn_alpha_th *= 0.9
                    elif "weak_trend" in mtf_regime:
                        dyn_alpha_th *= 0.95

                al = max(0, min(al, 2))
                as_ = max(0, min(as_, 2))
                best_alpha = max(al, as_)

                if best_alpha < dyn_alpha_th:
                    continue

                long_sig = al >= as_
                if htf_dir > 0 and not long_sig:
                    continue
                if htf_dir < 0 and long_sig:
                    continue

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

                # Risk
                dyn_risk = P["base_risk"]
                if isinstance(htf_regime, str) and "strong_trend" in htf_regime:
                    dyn_risk *= 1.3
                elif isinstance(htf_regime, str) and "weak_trend" in htf_regime:
                    dyn_risk *= 1.1

                risk_amt = cash * dyn_risk
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
            "win_rate": wins / trades * 100 if trades > 0 else 0,
            "max_dd": max_dd
        }


############################################################
# 9. Backtest Runner
############################################################
def run_backtest(symbols: List[str], days: int, capital: float,
                 seed: Optional[int], source: str):
    logger.info("🚀 SmartBacktest V11_1 运行中...")
    logger.info("🪙 币种: %s", symbols)

    market = RealMarketData()
    engine = AdaptiveSignalEngineV11()
    results = {}

    each_cap = capital / len(symbols)

    for sym in symbols:
        logger.info("🔍 处理 %s", sym)

        try:
            if source == "real":
                df_ltf = market.get_recent_klines(sym, "5m", days)
                df_mtf = market.get_recent_klines(sym, "1h", days + 3)
                df_htf = market.get_recent_klines(sym, "4h", days + 7)
                print(f"📥 {sym} 5m={len(df_ltf)}, 1h={len(df_mtf)}, 4h={len(df_htf)}")
            else:
                df_ltf = generate_mock_data(sym, days, seed)
                df_mtf = df_ltf.resample("1H").agg({
                    "open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum"
                }).dropna()
                df_htf = df_ltf.resample("4H").agg({
                    "open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum"
                }).dropna()

            res = engine.run(sym, df_ltf, df_mtf, df_htf, each_cap)
            results[sym] = res

        except Exception as e:
            logger.error("❌ %s 失败: %s", sym, e)

    total_pnl = sum([r["pnl"] for r in results.values()])
    total_trades = sum([r["trades"] for r in results.values()])
    total_wins = sum([r["wins"] for r in results.values()])
    win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
    max_dd = min([r["max_dd"] for r in results.values()]) if results else 0

    print("\n========== 📈 SmartBacktest V11_1 报告 ==========")
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


############################################################
# MAIN
############################################################
def parse_symbols(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    pa.add_argument("--days", type=int, default=30)
    pa.add_argument("--initial-capital", type=float, default=10000)
    pa.add_argument("--seed", type=int, default=None)
    pa.add_argument("--data-source", type=str, default="real")
    args = pa.parse_args()

    run_backtest(
        parse_symbols(args.symbols),
        args.days,
        capital=args.initial_capital,
        seed=args.seed,
        source=args.data_source,
    )


if __name__ == "__main__":
    main()
