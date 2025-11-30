#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SmartBacktest v10
====================================
第二季 · 稳定版内测：
- Regime-Switch 市场状态切换
- Anti-Noise 强降噪机制
- BTC / ETH 专用参数自适应
- 交易频率门控（min bars between trades）
- 延续 V9_2 的 Alpha 方向引擎（趋势 + 结构主导）
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
# 1. 价格结构：HH/HL/LH/LL + 假突破 + 震荡检测
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
        # 结构方向：简单 3-K 结构判定
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

        # 震荡区：连续若干根实体都很小
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
# 2. 趋势 / ATR 等基础指标
# ============================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # 均线趋势
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

    # EMA20 斜率 → 趋势强度 [0,1]
    ema20 = d["close"].ewm(span=20).mean()
    slope = (ema20 - ema20.shift(5)) / (ema20.shift(5).abs() + 1e-9)
    slope = (slope.abs() * 10).clip(0, 1)
    d["trend_strength"] = slope

    return d


# ============================================================
# 3. 策略信号：MACD / EMA / Turtle / Boll / Breakout
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

    # Turtle 通道
    hh = d["high"].rolling(20).max()
    ll = d["low"].rolling(20).min()
    d["sig_turtle"] = np.where(
        close > hh.shift(1),
        1,
        np.where(close < ll.shift(1), -1, 0),
    )

    # Boll 收敛/扩散
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
# 4. ensemble（只做共识强度，不做方向）
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
# 5. Regime + Alpha：趋势 + 结构主导 + 降噪
# ============================================================
def compute_regime_and_alpha(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    close = d["close"].values
    atr = d["atr"].values
    trend_strength = d["trend_strength"].values
    struct_strength = d["structure_strength"].values
    consensus = d["consensus_strength"].values
    fake = d["fake_break"].values
    chop = d["chop"].values
    struct = d["structure"].values

    n = len(d)
    vol_level = np.zeros(n)
    regime = np.array(["unknown"] * n, dtype=object)
    alpha_long = np.zeros(n)
    alpha_short = np.zeros(n)

    # 计算 ATR 的滚动平均，用来判断高/低波动
    atr_series = pd.Series(atr)
    atr_mean = atr_series.rolling(100, min_periods=20).mean().values

    for i in range(n):
        c = close[i]
        a = atr[i]
        if c > 0 and not np.isnan(a):
            v = a / c * 200.0  # 大致 0~若干
        else:
            v = 0.0
        v = max(0.0, min(v, 2.0))  # 不做太夸张
        vol_level[i] = v

        # 基于 ATR 相对均值判断高/低波动
        if atr_mean[i] > 0:
            vol_ratio = a / atr_mean[i]
        else:
            vol_ratio = 1.0

        # Regime 判定：
        # 优先级：choppy > low_vol > high_vol > strong_trend > weak_trend > range
        if chop[i] > 0.5:
            reg = "choppy"
        elif vol_ratio < 0.5:
            reg = "low_vol"
        elif vol_ratio > 1.8:
            reg = "high_vol"
        elif trend_strength[i] > 0.6 and 0.2 <= v <= 1.2:
            reg = "strong_trend"
        elif trend_strength[i] > 0.3 and 0.1 <= v <= 1.5:
            reg = "weak_trend"
        else:
            reg = "range"

        regime[i] = reg

        # --- Alpha 基础评分 ---
        base_long = (
            0.45 * trend_strength[i]
            + 0.35 * struct_strength[i]
            + 0.15 * consensus[i]
            + 0.05 * (1.0 - abs(v - 0.8))  # 中高波动略偏好
        )
        base_short = (
            0.45 * trend_strength[i]
            + 0.35 * struct_strength[i]
            + 0.15 * consensus[i]
            + 0.05 * (1.0 - abs(v - 0.8))
        )

        # 结构方向加成
        if struct[i] in ("HH", "HL"):
            base_long += 0.20
        if struct[i] in ("LL", "LH"):
            base_short += 0.20

        # 假突破惩罚
        if fake[i] > 0.5:
            base_long *= 0.7
            base_short *= 0.7

        # 震荡惩罚
        if reg == "choppy":
            base_long *= 0.5
            base_short *= 0.5

        # 极低波动惩罚
        if reg == "low_vol":
            base_long *= 0.4
            base_short *= 0.4

        # 适度强化趋势 Regime
        if reg == "strong_trend":
            base_long *= 1.1
            base_short *= 1.1
        elif reg == "weak_trend":
            base_long *= 1.05
            base_short *= 1.05

        alpha_long[i] = min(max(base_long, 0.0), 1.5)
        alpha_short[i] = min(max(base_short, 0.0), 1.5)

    d["vol_level"] = vol_level
    d["regime"] = regime
    d["alpha_long"] = alpha_long
    d["alpha_short"] = alpha_short

    return d


# ============================================================
# 6. 自适应交易引擎 V10（Regime-Switch + Anti-Noise）
# ============================================================
class AdaptiveSignalEngineV10:
    def __init__(self):
        # 全局基础参数
        self.base = {
            "sl_atr_mult": 1.8,
            "tp_atr_mult": 3.2,
            "trail_atr_mult": 1.8,
            "min_rr": 1.1,
            "alpha_threshold": 0.60,
            "base_risk": 0.01,
            "cooldown_bars": 12 * 12,      # 连续亏损冷静期
            "max_loss_streak": 3,
            "min_bars_between_trades": 10, # 交易频率门控
        }
        # 重点：BTC / ETH 专用参数（自适应）
        self.override = {
            # BTC 噪音多 → 提高 alpha 阈值、放大止损倍率、降低单笔风险
            "BTC": {
                "base_risk": 0.005,
                "alpha_threshold": 0.70,
                "sl_atr_mult": 2.5,
                "tp_atr_mult": 5.0,
                "min_bars_between_trades": 15,
            },
            # ETH 趋势更清晰 → 略放宽
            "ETH": {
                "base_risk": 0.006,
                "alpha_threshold": 0.60,
                "sl_atr_mult": 2.2,
                "tp_atr_mult": 4.2,
                "min_bars_between_trades": 10,
            },
        }

    def get_params(self, symbol: str) -> Dict[str, float]:
        P = self.base.copy()
        usym = symbol.upper()
        for k, v in self.override.items():
            if k in usym:
                P.update(v)
        return P

    def run(self, symbol: str, df: pd.DataFrame, capital: float) -> Dict[str, float]:
        P = self.get_params(symbol)

        if df is None or len(df) == 0:
            logger.warning("%s 数据为空，跳过回测", symbol)
            return {"pnl": 0.0, "trades": 0, "wins": 0, "win_rate": 0.0, "max_dd": 0.0}

        # 指标 & 因子流水线
        d = compute_indicators(df)
        d = compute_market_structure(d)
        d = compute_strategy_signals(d)
        d = compute_ensemble(d)
        d = compute_regime_and_alpha(d)

        cash = capital
        pos = 0          # 1=多，-1=空，0=空仓
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

        last_entry_bar: Optional[int] = None  # 频率门控使用

        for bar_index, (ts, row) in enumerate(d.iterrows()):
            price = float(row["close"])
            atr = float(row["atr"]) if not np.isnan(row["atr"]) else 0.0
            regime = str(row["regime"])
            chop = float(row["chop"])
            fake_break = float(row["fake_break"])
            consensus = float(row["consensus_strength"])
            alpha_long = float(row["alpha_long"])
            alpha_short = float(row["alpha_short"])

            # ------------------ 持仓管理 ------------------
            if pos != 0:
                # Trailing Stop
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

            # ------------------ 更新权益 / 回撤 ------------------
            if pos != 0:
                eq = cash + (price - entry) * size * pos
            else:
                eq = cash

            max_eq = max(max_eq, eq)
            if max_eq > 0:
                dd = (eq - max_eq) / max_eq * 100.0
                max_dd = min(max_dd, dd)

            # ------------------ 空仓 → 是否开新仓 ------------------
            if pos == 0:
                # 冷静期
                if cooldown > 0:
                    cooldown -= 1
                    continue

                # ATR 无效不做
                if atr <= 0:
                    continue

                # 交易频率门控：至少间隔 N 根 K
                if last_entry_bar is not None:
                    if bar_index - last_entry_bar < P["min_bars_between_trades"]:
                        continue

                # Regime 降噪：在 choppy / low_vol 直接禁止开仓
                if regime in ("choppy", "low_vol"):
                    continue

                # 假突破 + 共识弱 → 过滤
                if fake_break > 0.5 and consensus < 0.6:
                    continue

                # 动态 alpha 阈值：趋势期略放宽，高波动期略收紧
                dyn_alpha_th = P["alpha_threshold"]
                if regime == "strong_trend":
                    dyn_alpha_th *= 0.9
                elif regime == "high_vol":
                    dyn_alpha_th *= 1.1
                elif regime == "range":
                    dyn_alpha_th *= 1.0
                elif regime == "weak_trend":
                    dyn_alpha_th *= 0.95

                alpha_long_c = max(0.0, min(alpha_long, 1.5))
                alpha_short_c = max(0.0, min(alpha_short, 1.5))
                best_alpha = max(alpha_long_c, alpha_short_c)

                if best_alpha < dyn_alpha_th:
                    continue

                long_sig = alpha_long_c >= alpha_short_c

                # 计算止损 / 止盈
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

                # Regime 调整风险：强趋势放大，range 稍缩，小心
                dyn_base_risk = P["base_risk"]
                if regime == "strong_trend":
                    dyn_base_risk *= 1.3
                elif regime == "weak_trend":
                    dyn_base_risk *= 1.1
                elif regime == "high_vol":
                    dyn_base_risk *= 0.8
                elif regime == "range":
                    dyn_base_risk *= 0.9

                risk_amt = cash * dyn_base_risk
                if risk_amt <= 0:
                    continue

                size = risk_amt / sl_dist
                if size <= 0:
                    continue

                # 开仓
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
# 7. 多币种回测入口
# ============================================================
def run_backtest(symbols: List[str], days: int, capital: float,
                 seed: Optional[int], source: str):
    logger.info("🚀 SmartBacktest V10 启动")
    logger.info("🪙 币种: %s", symbols)
    logger.info("📅 回测天数: %d", days)
    logger.info("📊 数据源: %s", source)

    market = RealMarketData()
    engine = AdaptiveSignalEngineV10()

    results: Dict[str, Dict[str, float]] = {}
    each_cap = capital / len(symbols) if symbols else capital

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

    print("\n========== 📈 SmartBacktest V10 报告 ==========")
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
    pa = argparse.ArgumentParser(description="SmartBacktest V10")
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
