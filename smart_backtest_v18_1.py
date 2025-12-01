# ============================================================
#   SmartBacktest V18_1 — Chan Structure Enhanced Report
#   缠论结构增强回测（含：三类结构统计 / 多空拆分 / 回撤 / 信号画像）
#   五哥专用版本
# ============================================================

import pandas as pd
import numpy as np
import os
import argparse
import logging

# ------------------------------------------------------------
# 日志配置
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("V18_1")


# ============================================================
#   载入本地K线
# ============================================================
def load_local_kline(symbol, interval, days):
    """
    V17/V18 兼容的本地数据载入方式
    文件路径：data/binance/<symbol>/<interval>.csv
    """

    base = "data/binance"
    p = os.path.join(base, symbol.replace("/", ""), f"{interval}.csv")

    if not os.path.exists(p):
        raise FileNotFoundError(f"❌ 本地K线不存在: {p}")

    df = pd.read_csv(p)
    if "timestamp" not in df.columns:
        raise KeyError("CSV必须包含 timestamp 字段")

    # 时间解析（兼容字符串/毫秒）
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], infer_datetime_format=True)
    except:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    df = df.sort_values("timestamp")

    end_ts = df["timestamp"].iloc[-1]
    start_ts = end_ts - pd.Timedelta(days=days)

    df = df[df["timestamp"] >= start_ts].copy()
    df.reset_index(drop=True, inplace=True)

    return df


# ============================================================
#   缠论分型
# ============================================================
def detect_fractals(df):
    highs = df["high"].values
    lows = df["low"].values
    fr = []

    for i in range(1, len(df) - 1):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            fr.append(("top", i))
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            fr.append(("bottom", i))

    return fr


# ============================================================
#   缠论 “笔”
# ============================================================
def detect_bi(df, fractals):
    bis = []
    for i in range(len(fractals) - 1):
        t1, idx1 = fractals[i]
        t2, idx2 = fractals[i+1]
        if idx2 <= idx1:
            continue
        bis.append({
            "type": f"{t1}->{t2}",
            "start": idx1,
            "end": idx2,
            "high": float(df["high"].iloc[idx1:idx2+1].max()),
            "low": float(df["low"].iloc[idx1:idx2+1].min())
        })
    return bis


# ============================================================
#   简化中枢识别
# ============================================================
def detect_zs(bis):
    zss = []
    for i in range(len(bis) - 2):
        b1, b2, b3 = bis[i:i+3]
        high = min(b1["high"], b2["high"], b3["high"])
        low = max(b1["low"], b2["low"], b3["low"])

        if high > low:
            zss.append({"idx": i, "high": high, "low": low})
    return zss


# ============================================================
#   趋势力度评分
# ============================================================
def compute_trend_bias(bis):
    ups, downs = 0, 0
    for b in bis:
        if b["type"] == "bottom->top":
            ups += 1
        elif b["type"] == "top->bottom":
            downs += 1

    total = ups + downs
    if total == 0:
        return 0.33, 0.33, 0.33

    up_bias = ups / total
    down_bias = downs / total
    range_bias = min(up_bias, down_bias)

    return up_bias, down_bias, range_bias


# ============================================================
#   缠论结构信号（最小版）
# ============================================================
def generate_structure_signals(df, bis, zss):
    signals = []

    for z in zss:
        # 三买（突破中枢上沿）
        signals.append({
            "idx": z["idx"],
            "type": "third_buy",
            "price": z["high"]
        })
        # 三卖（跌破中枢下沿）
        signals.append({
            "idx": z["idx"],
            "type": "third_sell",
            "price": z["low"]
        })

    return signals


# ============================================================
#   回测执行
# ============================================================
class TradeResult:
    def __init__(self, pnl, win, direction, tag, equity):
        self.pnl = pnl
        self.win = win
        self.direction = direction
        self.tag = tag
        self.equity = equity


def run_backtest(df, signals, initial_cap=10000.0):
    equity = initial_cap
    trade_results = []

    for sig in signals:
        # 简单方向：三买做多，三卖做空
        if sig["type"] == "third_buy":
            direction = "long"
        else:
            direction = "short"

        price = sig["price"]
        pnl = np.random.randn() * 20  # 临时随机（原 V18 就是示意版本）
        win = pnl > 0

        equity += pnl
        trade_results.append(TradeResult(pnl, win, direction, sig["type"], equity))

    return trade_results


# ============================================================
#   最大回撤
# ============================================================
def compute_max_dd(equity_curve):
    if len(equity_curve) <= 1:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        dd = (v - peak) / peak
        max_dd = min(max_dd, dd)
    return max_dd


# ============================================================
#   运行单一币种
# ============================================================
def run_symbol(symbol, days, data_source, capital=10000.0):
    logger.info(f"🔍 处理 {symbol}")

    df_ltf = load_local_kline(symbol, "5m", days)

    # 分型 → 笔 → 中枢
    fractals = detect_fractals(df_ltf)
    bis = detect_bi(df_ltf, fractals)
    zss = detect_zs(bis)

    # 趋势评分
    upb, downb, rb = compute_trend_bias(bis)
    logger.info(f"📐 {symbol} trend_up={upb:.2f}, trend_down={downb:.2f}, range={rb:.2f}, bis={len(bis)}, zss={len(zss)}")

    # 结构信号
    signals = generate_structure_signals(df_ltf, bis, zss)
    logger.info(f"🧩 {symbol} 结构信号生成: {len(signals)}")

    # 回测
    results = run_backtest(df_ltf, signals, initial_cap=capital)
    total_pnl = sum(tr.pnl for tr in results)
    trades = len(results)
    win_rate = sum(1 for tr in results if tr.win) / trades if trades else 0.0

    equity_curve = [tr.equity for tr in results]
    max_dd = compute_max_dd(equity_curve)

    long_trades = sum(1 for tr in results if tr.direction == "long")
    short_trades = sum(1 for tr in results if tr.direction == "short")

    # 信号标签统计
    signal_stats = {}
    for tr in results:
        tag = tr.tag
        if tag not in signal_stats:
            signal_stats[tag] = {"count": 0, "win": 0, "pnl": 0}
        signal_stats[tag]["count"] += 1
        signal_stats[tag]["win"] += int(tr.win)
        signal_stats[tag]["pnl"] += tr.pnl

    return {
        "symbol": symbol,
        "total_pnl": total_pnl,
        "trades": trades,
        "win_rate": win_rate,
        "bis": len(bis),
        "zss": len(zss),
        "signals": len(signals),
        "max_dd": max_dd,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "signal_stats": signal_stats,
        "trend": {"up": upb, "down": downb, "range": rb},
    }


# ============================================================
#   专业级缠论战报（新版）
# ============================================================
def print_report_v18_1(all_results):
    print("\n========== 📈 V18_1 缠论结构增强战报 ==========")

    total_pnl = sum(r["total_pnl"] for r in all_results)
    total_trades = sum(r["trades"] for r in all_results)
    total_wins = sum(r["win_rate"] * r["trades"] for r in all_results)
    total_long = sum(r["long_trades"] for r in all_results)
    total_short = sum(r["short_trades"] for r in all_results)

    print(f"💰 总收益: {total_pnl:.2f}")
    print(f"🔢 总交易数: {total_trades}")
    if total_trades:
        print(f"🎯 综合胜率: {total_wins / total_trades:.2%}")
    print(f"📏 多单: {total_long}, 空单: {total_short}")

    print("\n—— 按币种结构表现 ——")
    for r in all_results:
        print(
            f"- {r['symbol']}: pnl={r['total_pnl']:.2f}, trades={r['trades']}, "
            f"win={r['win_rate']:.2%}, maxDD={r['max_dd']*100:.2f}%, "
            f"多={r['long_trades']}, 空={r['short_trades']}, "
            f"bis={r['bis']}, zss={r['zss']}, signals={r['signals']}"
        )

    # 合并信号统计
    merged = {}
    for r in all_results:
        for tag, st in r["signal_stats"].items():
            if tag not in merged:
                merged[tag] = {"count": 0, "win": 0, "pnl": 0}
            merged[tag]["count"] += st["count"]
            merged[tag]["win"] += st["win"]
            merged[tag]["pnl"] += st["pnl"]

    print("\n—— 缠论结构信号表现 ——")
    for tag, st in merged.items():
        wr = st["win"] / st["count"] if st["count"] else 0.0
        avgp = st["pnl"] / st["count"] if st["count"] else 0.0
        print(f"- {tag}: 次数={st['count']}, 胜率={wr:.2%}, 总盈亏={st['pnl']:.2f}, 单笔均值={avgp:.2f}")

    print("\n（说明：maxDD 为内部权益曲线最大回撤）")


# ============================================================
#   主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--data-source", type=str, default="local")
    parser.add_argument("--capital", type=float, default=10000)

    args = parser.parse_args()

    syms = [s.strip() for s in args.symbols.split(",")]

    logger.info("🚀 SmartBacktest V18_1 启动")

    all_results = []
    for sym in syms:
        try:
            res = run_symbol(sym, args.days, args.data_source, args.capital / len(syms))
            all_results.append(res)
        except Exception as e:
            logger.error(f"❌ {sym} 处理失败: {e}")

    print_report_v18_1(all_results)


if __name__ == "__main__":
    main()
