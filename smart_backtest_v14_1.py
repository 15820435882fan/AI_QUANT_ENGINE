# ===========================================================
#   SmartBacktest V14_1 — Regime 修复版 + 权重引擎修复版
#   by 小超人，专为五哥量化体系打造
# ===========================================================

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path

# -----------------------------------------------------------
# Logger 设置
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# -----------------------------------------------------------
# 工具函数
# -----------------------------------------------------------
def load_local_kline(symbol, interval, days):
    """从本地缓存加载数据"""
    base = Path("data/binance")
    folder = base / symbol.replace("/", "")
    file = folder / f"{interval}.csv"

    if not file.exists():
        raise FileNotFoundError(f"文件不存在: {file}")

    df = pd.read_csv(file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df = df.sort_index()

    # 截取最近 N 天
    df = df.iloc[-days * (1440 // int(interval.replace("m", ""))) :]
    return df


# -----------------------------------------------------------
# 计算趋势 slope
# -----------------------------------------------------------
def calc_slope(series, n=20):
    if len(series) < n:
        return 0
    y = series[-n:].values
    x = np.arange(n)
    slope = np.polyfit(x, y, 1)[0] / y.mean()
    return slope


# -----------------------------------------------------------
# 计算趋势分数（修复版）
# -----------------------------------------------------------
def compute_trend_score(df_mtf, df_htf):
    mtf_slope = calc_slope(df_mtf["close"], 48)    # 2 天
    htf_slope = calc_slope(df_htf["close"], 30)    # 5 天

    # 修复关键问题：不要过度放大
    trend_raw = abs(mtf_slope * 1000) + abs(htf_slope * 2000)
    trend_score = min(1.0, trend_raw)

    return trend_score


# -----------------------------------------------------------
# 计算震荡分数（修复版）
# -----------------------------------------------------------
def compute_range_score(df_mtf):
    close = df_mtf["close"]
    high = df_mtf["high"]
    low = df_mtf["low"]

    # BOLL 宽度
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = mid + std
    lower = mid - std
    boll_width = ((upper - lower) / close).iloc[-1]

    # ATR 比例
    hl = high - low
    atr = hl.rolling(20).mean().iloc[-1]
    atr_ratio = (atr / close.iloc[-1])

    # 修复过度放大
    bw_norm = min(1.0, boll_width * 40)
    atr_norm = min(1.0, atr_ratio * 120)

    # range ↑ when band wider & ATR lower
    range_score = (bw_norm + (1 - atr_norm)) / 2
    range_score = max(0, min(1, range_score))

    return range_score


# -----------------------------------------------------------
# Regime 判定（修复版）
# -----------------------------------------------------------
def classify_regime(trend_score, range_score):
    if trend_score > 0.55 and range_score < 0.45:
        return "trend"
    elif range_score > 0.60:
        return "range"
    else:
        return "mixed"


# -----------------------------------------------------------
# 权重引擎（修复版）
# -----------------------------------------------------------
def compute_weight(symbol, trend_score, range_score, pnl_history):
    # ETH 震荡更大，需要降低趋势分数
    if symbol == "ETH/USDT":
        trend_score *= 0.7

    # 根据 regime 强弱赋权
    base = trend_score * 0.6 + (1 - range_score) * 0.4

    # 引入 PnL 表现（防止 ETH 持续拖累）
    if len(pnl_history) > 10:
        pnl_factor = np.tanh(np.mean(pnl_history[-10:]) / 20)
        base = base * (1 + pnl_factor)

    return max(0.05, min(0.95, base))


# -----------------------------------------------------------
# 冷静期引擎（修复版）
# -----------------------------------------------------------
def calc_cooldown_bars(regime, range_score):
    if regime == "trend":
        return int(30 + range_score * 40)
    elif regime == "range":
        return int(20 + range_score * 60)
    return 25


# -----------------------------------------------------------
# 策略执行引擎（精简示例）
# -----------------------------------------------------------
class MultiEngine:
    def __init__(self):
        self.position = None
        self.entry_price = 0
        self.pnl_history = []
        self.cooldown = 0

    def step(self, symbol, row, regime, weight):
        price = row.close
        result = {"pnl": 0, "trade": None}

        if self.cooldown > 0:
            self.cooldown -= 1
            return result

        # ----------------------------------------
        # trend 策略
        # ----------------------------------------
        if regime == "trend":
            if self.position is None:
                self.position = "long"
                self.entry_price = price
                result["trade"] = "buy"

            else:
                if price < self.entry_price * 0.985:
                    pnl = price - self.entry_price
                    self.pnl_history.append(pnl)
                    result["pnl"] = pnl
                    self.position = None
                    cooldown = calc_cooldown_bars("trend", 0.3)
                    self.cooldown = cooldown
                    result["trade"] = "sell"

        # ----------------------------------------
        # range 策略
        # ----------------------------------------
        elif regime == "range":
            if self.position is None and row.low < row.close * 0.996:
                self.position = "long"
                self.entry_price = price
                result["trade"] = "buy"

            else:
                if price > self.entry_price * 1.003:
                    pnl = price - self.entry_price
                    self.pnl_history.append(pnl)
                    result["pnl"] = pnl
                    self.position = None
                    cooldown = calc_cooldown_bars("range", 0.6)
                    self.cooldown = cooldown
                    result["trade"] = "sell"

        return result


# -----------------------------------------------------------
# 主回测
# -----------------------------------------------------------
def run(symbol, df_ltf, df_mtf, df_htf):
    engine = MultiEngine()
    results = []

    trend_score = compute_trend_score(df_mtf, df_htf)
    range_score = compute_range_score(df_mtf)
    regime = classify_regime(trend_score, range_score)
    weight = compute_weight(symbol, trend_score, range_score, engine.pnl_history)

    logging.info(
        f"📊 {symbol} regime={regime}, trend={trend_score:.2f}, range={range_score:.2f}, weight={weight:.2f}"
    )

    for ts, row in df_ltf.iterrows():
        res = engine.step(symbol, row, regime, weight)
        results.append(res["pnl"])

    return sum(results), len(results), regime, weight


# -----------------------------------------------------------
# 主入口
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--data-source", type=str, default="local")
    args = parser.parse_args()

    symbols = args.symbols.split(",")

    logging.info("🚀 SmartBacktest V14_1 启动")

    total_pnl = 0
    total_trades = 0

    for sym in symbols:
        df_ltf = load_local_kline(sym, "5m", args.days)
        df_mtf = load_local_kline(sym, "1h", args.days + 3)
        df_htf = load_local_kline(sym, "4h", args.days + 7)

        pnl, trades, regime, weight = run(sym, df_ltf, df_mtf, df_htf)

        logging.info(
            f"{sym}: pnl={pnl:.2f}, trades={trades}, regime={regime}, weight={weight:.2f}"
        )

        total_pnl += pnl
        total_trades += trades

    logging.info("========== 📈 SmartBacktest V14_1 报告 ==========")
    logging.info(f"总收益: {total_pnl:.2f}")
    logging.info(f"总交易数: {total_trades}")


if __name__ == "__main__":
    main()
