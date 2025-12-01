# ============================================================
#                 SmartBacktest V18_2_fixed (Full)
#       —— 修复索引错误 + 完整缠论结构引擎 + 战报增强
# ============================================================

import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ============================================================
#                  数据读取（本地CSV修复版）
# ============================================================

def load_local_kline(symbol: str, interval: str, days: int):
    path = f"data/binance/{symbol.replace('/', '')}/{interval}.csv"
    try:
        df = pd.read_csv(path)

        # ★★ 最关键修复：必须 reset_index
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # 只取最近N天（用于5m）
        bars = days * (1440 // 5)
        df = df.tail(bars).reset_index(drop=True)

        logging.info(
            f"📥 [Local] 载入 {symbol} {interval}, 行数={len(df)}, 天数≈{days}"
        )
        return df
    except Exception as e:
        logging.error(f"❌ 加载数据失败: {symbol} {interval}, {e}")
        return None


# ============================================================
#               缠论分型（必须基于连续索引）
# ============================================================

def detect_fractals(df):
    highs = df["high"].values
    lows = df["low"].values

    up_fractals = []
    down_fractals = []

    for i in range(2, len(df) - 2):
        # 顶分型
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            up_fractals.append(i)
        # 底分型
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            down_fractals.append(i)

    return up_fractals, down_fractals


# ============================================================
#                           缠论笔
# ============================================================

class Bi:
    def __init__(self, start, end, direction):
        self.start = start
        self.end = end
        self.direction = direction  # up / down


def detect_bi(df, up_f, down_f):
    bis = []
    f = sorted(up_f + down_f)
    highs = df["high"].values
    lows = df["low"].values

    for i in range(2, len(f)):
        a, b, c = f[i - 2], f[i - 1], f[i]

        # 上升笔（低 -> 高）
        if a in down_f and b in up_f:
            if highs[b] > highs[a] and highs[b] > highs[c]:
                bis.append(Bi(a, b, "up"))

        # 下降笔（高 -> 低）
        if a in up_f and b in down_f:
            if lows[b] < lows[a] and lows[b] < lows[c]:
                bis.append(Bi(a, b, "down"))

    return bis


# ============================================================
#                     三买三卖结构识别
# ============================================================

class StructureSignal:
    def __init__(self, index, kind):
        self.index = index
        self.kind = kind  # third_buy / third_sell


def detect_third_signals(df, bis):
    signals = []
    highs = df["high"].values
    lows = df["low"].values

    for i in range(2, len(bis)):
        b1, b2, b3 = bis[i - 2], bis[i - 1], bis[i]

        # 三买：下-上-下，且b3高点 > b1高点
        if b1.direction == "down" and b2.direction == "up" and b3.direction == "down":
            if highs[b3.end] > highs[b1.end]:
                signals.append(StructureSignal(b3.end, "third_buy"))

        # 三卖：上-下-上，且b3低点 < b1低点
        if b1.direction == "up" and b2.direction == "down" and b3.direction == "up":
            if lows[b3.end] < lows[b1.end]:
                signals.append(StructureSignal(b3.end, "third_sell"))

    return signals


# ============================================================
#                         趋势过滤
# ============================================================

def compute_trend(df):
    close = df["close"]
    ma20 = close.rolling(20).mean()
    up = (close > ma20).mean()
    down = (close < ma20).mean()
    rng = 1 - abs(up - down)
    return up, down, rng


# ============================================================
#                    回测：结构止损 + 动态RR
# ============================================================

def compute_ATR(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()


def backtest_structure(df, signals):

    df["atr"] = compute_ATR(df, 14)

    pnl = 0
    trades = 0
    wins = 0

    rr_target = 2.4          # 动态RR
    atr_mult_stop = 1.6      # 基于结构的波动止损
    min_stop_pct = 0.0035
    max_bars = 350
    min_spacing = 12

    last_entry_idx = None

    for sig in signals:

        if last_entry_idx and sig.index - last_entry_idx < min_spacing:
            continue
        last_entry_idx = sig.index

        entry = df.loc[sig.index, "close"]
        atr = df.loc[sig.index, "atr"]

        stop = atr * atr_mult_stop
        stop = max(stop, entry * min_stop_pct)

        if sig.kind == "third_buy":
            sl = entry - stop
            tp = entry + stop * rr_target
        else:
            sl = entry + stop
            tp = entry - stop * rr_target

        exit_price = None
        for j in range(sig.index + 1, min(sig.index + max_bars, len(df))):
            high = df.loc[j, "high"]
            low = df.loc[j, "low"]

            # 止损
            if sig.kind == "third_buy" and low <= sl:
                exit_price = sl
                break
            if sig.kind == "third_sell" and high >= sl:
                exit_price = sl
                break

            # 止盈
            if sig.kind == "third_buy" and high >= tp:
                exit_price = tp
                break
            if sig.kind == "third_sell" and low <= tp:
                exit_price = tp
                break

        if exit_price is None:
            exit_price = df.loc[
                min(sig.index + max_bars - 1, len(df) - 1),
                "close"
            ]

        trades += 1

        if sig.kind == "third_buy":
            profit = exit_price - entry
        else:
            profit = entry - exit_price

        if profit > 0:
            wins += 1

        pnl += profit

    return pnl, trades, wins


# ============================================================
#                     主流程
# ============================================================

def run_symbol(symbol, days):

    df = load_local_kline(symbol, "5m", days)
    if df is None or len(df) < 500:
        logging.error(f"❌ {symbol} 数据不足")
        return None

    up_f, down_f = detect_fractals(df)
    bis = detect_bi(df, up_f, down_f)
    signals = detect_third_signals(df, bis)

    up, down, rng = compute_trend(df)

    # 趋势过滤
    signals = [
        s for s in signals
        if ((s.kind == "third_buy" and up > down * 1.05) or
            (s.kind == "third_sell" and down > up * 1.05))
    ]

    pnl, trades, wins = backtest_structure(df, signals)

    logging.info(
        f"📊 {symbol}: pnl={pnl:.2f}, trades={trades}, win={wins/trades if trades else 0:.2f}, "
        f"bis={len(bis)}, signals={len(signals)}, trend_up={up:.2f}, trend_down={down:.2f}"
    )

    return dict(
        symbol=symbol,
        pnl=pnl,
        trades=trades,
        wins=wins,
        bis=len(bis),
        signals=len(signals),
        trend_up=up,
        trend_down=down,
        trend_range=rng
    )


# ============================================================
#                           主入口
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--data-source", type=str, default="local")
    args = parser.parse_args()

    syms = [s.strip() for s in args.symbols.split(",")]

    total_pnl = 0
    total_trades = 0
    total_wins = 0

    print("")

    for sym in syms:
        res = run_symbol(sym, args.days)
        if res:
            total_pnl += res["pnl"]
            total_trades += res["trades"]
            total_wins += res["wins"]

    print("\n========== 📈 V18_2_fixed 缠论增强版 战报 ==========")
    print(f"💰 总收益: {total_pnl:.2f}")
    print(f"🔢 总交易数: {total_trades}")
    print(f"🎯 综合胜率: {total_wins/max(total_trades,1):.2f}")

