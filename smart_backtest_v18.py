# ==========================================
# smart_backtest_v18.py
# 完整版 — 缠论五层结构引擎（分型→笔→线段→中枢→三买）
# ==========================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ------------------------
# 工具函数
# ------------------------
def load_local_kline(symbol, interval, days):
    """
    载入本地K线
    """
    path = f"data/binance/{symbol.replace('/', '')}/{interval}.csv"
    df = pd.read_csv(path)

    # timestamp 处理
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        except:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp")
    df = df.reset_index(drop=True)

    if len(df) > 2000:
        df = df.iloc[-2000:]

    return df


# -------------------------
# 分型（顶分型、底分型）
# -------------------------
def detect_fractals(df):
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    fractals = np.zeros(n)

    for i in range(2, n-2):
        # 顶分型
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            fractals[i] = 1
        # 底分型
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            fractals[i] = -1

    return fractals
# ==========================================
# 笔 — 完整缠论逻辑
# ==========================================

class Bi:
    def __init__(self, start, end, high, low, direction):
        self.start = start
        self.end = end
        self.high = high
        self.low = low
        self.direction = direction  # up / down

def build_bi(df, fractals):
    """
    缠论严格笔构建
    """
    n = len(df)
    bis = []

    # 找所有分型点
    idxs = np.where(fractals != 0)[0]

    # 必须高低点交替
    valid_points = []
    last_type = 0

    for i in idxs:
        f = fractals[i]
        if f == last_type:
            continue
        valid_points.append(i)
        last_type = f

    # 构建笔
    for i in range(1, len(valid_points)):
        s = valid_points[i-1]
        e = valid_points[i]

        high = df["high"].iloc[s:e+1].max()
        low = df["low"].iloc[s:e+1].min()

        direction = "up" if df["close"].iloc[e] > df["close"].iloc[s] else "down"

        bis.append(Bi(s, e, high, low, direction))

    return bis
# ==========================================
# 线段（段）& 中枢
# ==========================================

class ZhongShu:
    def __init__(self, start_bi, end_bi, high, low):
        self.start_bi = start_bi
        self.end_bi = end_bi
        self.high = high
        self.low = low

def detect_zhongshu(bis):
    """
    中枢识别（严格三笔重叠）
    """
    zss = []

    for i in range(2, len(bis)):
        b1, b2, b3 = bis[i-2], bis[i-1], bis[i]

        high = min(b1.high, b2.high, b3.high)
        low = max(b1.low, b2.low, b3.low)

        if low <= high:  # 发生重叠
            zss.append(ZhongShu(i-2, i, high, low))

    return zss
# ==========================================
# 三类买点 / 卖点（V18 核心）
# ==========================================

class Signal:
    def __init__(self, ts_idx, direction, price):
        self.ts_idx = ts_idx
        self.direction = direction
        self.price = price

def detect_signals(df, bis, zss):
    """
    三买 / 三卖信号
    """
    signals = []

    for zs in zss:
        # 中枢离开
        if zs.end_bi + 2 >= len(bis):
            continue

        bi_leave = bis[zs.end_bi + 1]
        bi_back  = bis[zs.end_bi + 2]

        # 三买：回抽不破中枢下沿
        if bi_leave.direction == "up":
            if bi_back.low > zs.low:
                price = df["close"].iloc[bi_back.end]
                signals.append(Signal(bi_back.end, "long", price))

        # 三卖：回抽不破中枢上沿
        if bi_leave.direction == "down":
            if bi_back.high < zs.high:
                price = df["close"].iloc[bi_back.end]
                signals.append(Signal(bi_back.end, "short", price))

    return signals
# ==========================================
# 回测：仅做三类买卖点
# ==========================================

def backtest(df, signals, rr=2.0, sl_ratio=0.01):
    pnl = 0
    trades = 0
    wins = 0

    for s in signals:
        entry = s.price
        direction = s.direction

        # 止损
        sl = entry * (1 - sl_ratio) if direction == "long" else entry * (1 + sl_ratio)
        tp = entry + rr * (entry - sl) if direction == "long" else entry - rr * (sl - entry)

        trades += 1

        for i in range(s.ts_idx+1, len(df)):
            high = df["high"].iloc[i]
            low = df["low"].iloc[i]

            if direction == "long":
                if low <= sl:
                    pnl -= (entry - sl)
                    break
                if high >= tp:
                    pnl += (tp - entry)
                    wins += 1
                    break
            else:
                if high >= sl:
                    pnl -= (sl - entry)
                    break
                if low <= tp:
                    pnl += (entry - tp)
                    wins += 1
                    break

    win_rate = wins / trades if trades > 0 else 0
    return pnl, trades, win_rate
# ==========================================
# 主流程
# ==========================================

import argparse

def run_symbol(sym, days):
    df = load_local_kline(sym, "5m", days)
    fractals = detect_fractals(df)
    bis = build_bi(df, fractals)
    zss = detect_zhongshu(bis)
    signals = detect_signals(df, bis, zss)

    pnl, trades, win_rate = backtest(df, signals)

    logging.info(f"📊 {sym}: pnl={pnl:.2f}, trades={trades}, win_rate={win_rate:.2f}, bis={len(bis)}, zss={len(zss)}, signals={len(signals)}")

    return pnl, trades, win_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--days", type=int, default=60)
    args = parser.parse_args()

    syms = [s.strip() for s in args.symbols.split(",")]
    total_pnl = 0
    total_trades = 0

    for sym in syms:
        pnl, trades, win_rate = run_symbol(sym, args.days)
        total_pnl += pnl
        total_trades += trades

    print("\n========== 📈 V18缠论回测报告 ==========")
    print(f"总收益: {total_pnl:.2f}")
    print(f"总交易数: {total_trades}")

if __name__ == "__main__":
    main()
