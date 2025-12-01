# ============================================================
#                 SmartBacktest V18_2 (Full)
#      —— 缠论三笔结构 + 趋势过滤 + 结构止损 + 动态RR
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
#                  数据读取（兼容本地CSV）
# ============================================================

def load_local_kline(symbol: str, interval: str, days: int):
    path = f"data/binance/{symbol.replace('/', '')}/{interval}.csv"
    try:
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.tail(days * (1440 // 5))  # 只保留最近 N 天（适合 5m）
        return df
    except Exception as e:
        logging.error(f"❌ 加载本地数据失败: {symbol} {interval}, {e}")
        return None

# ============================================================
#                  缠论分型识别
# ============================================================

def detect_fractals(df):
    highs, lows = df["high"], df["low"]
    up_idx, down_idx = [], []
    for i in range(2, len(df) - 2):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            up_idx.append(i)
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            down_idx.append(i)
    return up_idx, down_idx

# ============================================================
#                  缠论“笔”识别
# ============================================================

class Bi:
    def __init__(self, start, end, direction):
        self.start = start
        self.end = end
        self.direction = direction  # up / down

def detect_bi(df, up_fractal, down_fractal):
    bis = []
    f = sorted(up_fractal + down_fractal)
    for i in range(2, len(f)):
        a, b, c = f[i - 2], f[i - 1], f[i]
        if a in up_fractal and b in down_fractal:
            if df["low"][b] < df["low"][a] and df["low"][b] < df["low"][c]:
                bis.append(Bi(a, b, "down"))
        if a in down_fractal and b in up_fractal:
            if df["high"][b] > df["high"][a] and df["high"][b] > df["high"][c]:
                bis.append(Bi(a, b, "up"))
    return bis

# ============================================================
#                三笔结构（三买三卖）
# ============================================================

class StructureSignal:
    def __init__(self, index, kind):
        self.index = index        # 触发点
        self.kind = kind          # third_buy / third_sell

def detect_third_signals(df, bis):
    signals = []
    for i in range(2, len(bis)):
        b1, b2, b3 = bis[i - 2], bis[i - 1], bis[i]

        # 三买：下-上-下，且 b3 高点 > b1 高点
        if b1.direction == "down" and b2.direction == "up" and b3.direction == "down":
            if df["high"][b3.end] > df["high"][b1.end]:
                signals.append(StructureSignal(b3.end, "third_buy"))

        # 三卖：上-下-上，且 b3 低点 < b1 低点
        if b1.direction == "up" and b2.direction == "down" and b3.direction == "up":
            if df["low"][b3.end] < df["low"][b1.end]:
                signals.append(StructureSignal(b3.end, "third_sell"))

    return signals

# ============================================================
#               趋势过滤（大方向）
# ============================================================

def compute_trend(df):
    ma20 = df["close"].rolling(20).mean()
    upward = (df["close"] > ma20).mean()
    downward = (df["close"] < ma20).mean()
    return upward, downward, 1 - abs(upward - downward)

# ============================================================
#               结构驱动的动态回测引擎
# ============================================================

def backtest_structure(
    df, signals, rr_target=2.5, atr_mult_stop=1.5, min_stop_pct=0.004,
    max_holding_bars=400, min_spacing_bars=10
):
    pnl = 0.0
    trades = 0
    wins = 0
    last_entry = None

    df["atr"] = compute_ATR(df, 14)

    for sig in signals:

        # 限制频率（跟冷静期不同，是“节奏控制”）
        if last_entry and sig.index - last_entry < min_spacing_bars:
            continue
        last_entry = sig.index

        entry = df.loc[sig.index, "close"]

        # 动态止损 = max(结构范围, 波动范围)
        sl = entry - max(df["atr"][sig.index] * atr_mult_stop, entry * min_stop_pct) if sig.kind == "third_buy" \
             else entry + max(df["atr"][sig.index] * atr_mult_stop, entry * min_stop_pct)

        tp = entry + (entry - sl) * rr_target if sig.kind == "third_buy" \
             else entry - (sl - entry) * rr_target

        # 模拟后续价格
        exit_price = None
        for j in range(sig.index + 1, min(sig.index + max_holding_bars, len(df))):
            high, low = df.loc[j, "high"], df.loc[j, "low"]

            # 先看止损
            if sig.kind == "third_buy" and low <= sl:
                exit_price = sl
                break
            if sig.kind == "third_sell" and high >= sl:
                exit_price = sl
                break

            # 再看止盈
            if sig.kind == "third_buy" and high >= tp:
                exit_price = tp
                break
            if sig.kind == "third_sell" and low <= tp:
                exit_price = tp
                break

        # 如果未出场，按最后一根 K 收盘
        if exit_price is None:
            exit_price = df.loc[min(sig.index + max_holding_bars - 1, len(df) - 1), "close"]

        trades += 1
        if (sig.kind == "third_buy" and exit_price > entry) or \
           (sig.kind == "third_sell" and exit_price < entry):
            wins += 1

        pnl += exit_price - entry if sig.kind == "third_buy" else entry - exit_price

    return pnl, trades, wins

# ============================================================
#               ATR 计算
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

# ============================================================
#               主流程
# ============================================================

def run_symbol(symbol, days):

    df = load_local_kline(symbol, "5m", days)
    if df is None or len(df) < 500:
        logging.error(f"❌ 无法载入 {symbol} 数据")
        return None

    up_f, down_f = detect_fractals(df)
    bis = detect_bi(df, up_f, down_f)
    signals = detect_third_signals(df, bis)

    up, down, rng = compute_trend(df)

    # 趋势过滤：只顺大方向
    signals = [
        s for s in signals
        if ((s.kind == "third_buy" and up > down * 1.05) or
            (s.kind == "third_sell" and down > up * 1.05))
    ]

    pnl, trades, wins = backtest_structure(df, signals)

    logging.info(
        f"📊 {symbol}: pnl={pnl:.2f}, trades={trades}, win_rate={wins/max(trades,1):.2f}, "
        f"bis={len(bis)}, zss=?, signals={len(signals)}"
    )

    return {
        "symbol": symbol,
        "pnl": pnl,
        "trades": trades,
        "wins": wins,
        "bis": len(bis),
        "signals": len(signals),
        "trend": (up, down, rng)
    }

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

    for sym in syms:
        res = run_symbol(sym, args.days)
        if res:
            total_pnl += res["pnl"]
            total_trades += res["trades"]

    print("\n========== 📈 V18_2 缠论结构增强版 - 回测结果 ==========")
    print(f"💰 总收益: {total_pnl:.2f}")
    print(f"🔢 总交易数: {total_trades}")
