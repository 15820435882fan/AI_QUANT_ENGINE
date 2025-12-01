import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ==================================
#  Local Data Loader
# ==================================
def load_local_kline(symbol: str, interval: str, days: int):
    """
    从本地 CSV 加载 K 线:
    data/binance/SYMBOL/interval.csv
    支持 timestamp 为:
    - 毫秒整数
    - 普通日期字符串 "2024-01-01 00:00:00"
    """
    sym = symbol.replace("/", "")
    path = f"data/binance/{sym}/{interval}.csv"

    try:
        df = pd.read_csv(path)
    except Exception as e:
        logging.error(f"❌ 文件不存在或读取失败: {path}, err={e}")
        return None

    # 统一小写列名
    df.columns = [c.lower() for c in df.columns]

    if "timestamp" not in df.columns:
        logging.error(f"❌ CSV 缺少 timestamp 列: {path}")
        return None

    ts = df["timestamp"]

    # 判断是数字(ms)还是字符串(datetime)
    if np.issubdtype(ts.dtype, np.number):
        # 纯数字，当成毫秒戳
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    else:
        # 字符串，让 pandas 自己解析
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        bad = df["timestamp"].isna().sum()
        if bad > 0:
            logging.warning(f"⚠️ {path} 有 {bad} 行 timestamp 解析失败，将被丢弃")
            df = df.dropna(subset=["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    # 截取最近 N 天
    total_minutes = days * 1440

    if interval.endswith("m"):
        step = int(interval.replace("m", ""))
        need = total_minutes // step
    elif interval.endswith("h"):
        step = int(interval.replace("h", "")) * 60
        need = total_minutes // step
    else:
        need = len(df)

    if need > 0 and len(df) > need:
        df = df.iloc[-need:].reset_index(drop=True)

    logging.info(
        f"📥 [LocalDataEngine] 载入本地数据: {symbol} {interval}, 天数={days}, 行数={len(df)}"
    )
    return df


# ==================================
#  Chan-style Fractals & Bi Segments
# ==================================

@dataclass
class BiSegment:
    start_index: int
    end_index: int
    direction: str          # "up" or "down"
    length_pct: float       # 相对涨跌幅
    bar_count: int          # 跨越 K 线根数
    high: float = None      # 这笔的最高价
    low: float = None       # 这笔的最低价


def detect_fractals(df: pd.DataFrame, left: int = 2, right: int = 2):
    """
    简单双边分型识别：
    - 顶分型：high[i] 为左右窗口最高点
    - 底分型：low[i] 为左右窗口最低点
    返回：fractal_list = [(idx, "top"), (idx, "bottom"), ...]
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    fractals = []

    for i in range(left, n - right):
        window_high = highs[i - left : i + right + 1].max()
        window_low = lows[i - left : i + right + 1].min()

        if highs[i] == window_high and highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            fractals.append((i, "top"))
        elif lows[i] == window_low and lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            fractals.append((i, "bottom"))

    return fractals


def detect_bis(df: pd.DataFrame, fractals, min_move_base: float = 0.001):
    """
    根据分型（顶/底）构造“笔”：
    - 相邻两个分型形成一笔
    - direction 由价格高低决定
    - 去掉涨跌幅过小的笔
    """
    closes = df["close"].values

    bis = []
    if len(fractals) < 2:
        return bis

    for i in range(1, len(fractals)):
        idx1, _t1 = fractals[i - 1]
        idx2, _t2 = fractals[i]
        if idx2 <= idx1:
            continue

        p1 = closes[idx1]
        p2 = closes[idx2]
        if p2 > p1:
            direction = "up"
        elif p2 < p1:
            direction = "down"
        else:
            continue

        length_pct = (p2 - p1) / p1
        bar_count = idx2 - idx1

        if abs(length_pct) < min_move_base:
            continue

        # high/low 先不算，后面统一从 df_ltf 里补
        bis.append(
            BiSegment(
                start_index=int(idx1),
                end_index=int(idx2),
                direction=direction,
                length_pct=float(length_pct),
                bar_count=int(bar_count),
            )
        )

    return bis


def filter_valid_bis(bis, min_bars: int = 7, min_move_pct: float = 0.003):
    """
    过滤出“有效笔”：
    - bar_count >= min_bars
    - |length_pct| >= min_move_pct
    """
    valid = []
    for bi in bis:
        if bi.bar_count >= min_bars and abs(bi.length_pct) >= min_move_pct:
            valid.append(bi)
    return valid


def compute_structure_factors(df_ltf: pd.DataFrame, df_mtf: pd.DataFrame, valid_bis):
    """
    结构因子 + MA 趋势因子 + 综合 regime 判定

    返回：
    {
        "struct_score": float(0~1),
        "struct_bias":  float(-1~1, 向上为正),
        "ma_score":     float(0~1),
        "final_score":  float(0~1),
        "regime":       "trend"/"mixed",
    }
    """
    # ---- 结构方向：看有效笔的长度和 ----
    up_len = sum(b.length_pct for b in valid_bis if b.direction == "up")
    down_len = sum(-b.length_pct for b in valid_bis if b.direction == "down")
    total_len = up_len + down_len

    if total_len <= 0:
        struct_bias = 0.0
        struct_score = 0.0
    else:
        struct_bias = (up_len - down_len) / total_len  # -1 ~ 1
        struct_score = min(1.0, abs(struct_bias) * 2.0)  # 放大一点

    # ---- MA 趋势因子：看 1h 的 EMA20/EMA60 + 斜率 ----
    if df_mtf is None or len(df_mtf) < 80:
        ma_score = 0.5
    else:
        close = df_mtf["close"].values
        ema_fast = pd.Series(close).ewm(span=20, adjust=False).mean().values
        ema_slow = pd.Series(close).ewm(span=60, adjust=False).mean().values
        spread = ema_fast - ema_slow
        slope = pd.Series(spread).diff().rolling(10).mean().iloc[-1]

        s = float(np.tanh(slope * 500))  # 压缩到 (-1,1)
        ma_score = 0.5 + 0.5 * s        # 映射到 0~1

    # ---- 综合打分 ----
    final_score = 0.5 * struct_score + 0.5 * abs(ma_score - 0.5) * 2.0

    regime = "trend" if final_score >= 0.6 else "mixed"

    return {
        "struct_score": float(struct_score),
        "struct_bias": float(struct_bias),
        "ma_score": float(ma_score),
        "final_score": float(final_score),
        "regime": regime,
    }


# ==================================
#  ATR & Structure Signals
# ==================================

def compute_atr(df: pd.DataFrame, period: int = 14):
    """
    标准 ATR 指标，用于设置止损/止盈 & 冷静期判断
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr


def generate_structure_signals(df_ltf: pd.DataFrame, valid_bis):
    """
    基于“缠论三笔结构 + 单笔趋势延伸”的结构信号引擎

    返回:
        long_signals:  set(bar_index)
        short_signals: set(bar_index)
    """
    long_signals = set()
    short_signals = set()

    if len(valid_bis) < 1:
        return long_signals, short_signals

    highs = df_ltf["high"].values
    lows = df_ltf["low"].values

    # --- 为每一笔补齐 high / low 属性 ---
    for bi in valid_bis:
        if bi.high is None:
            bi.high = float(highs[bi.start_index : bi.end_index + 1].max())
        if bi.low is None:
            bi.low = float(lows[bi.start_index : bi.end_index + 1].min())

    # --- 时间顺序排序 ---
    valid_bis = sorted(valid_bis, key=lambda b: b.start_index)

    # --- 三笔结构检查 ---
    for i in range(2, len(valid_bis)):
        b1, b2, b3 = valid_bis[i - 2], valid_bis[i - 1], valid_bis[i]
        d1, d2, d3 = b1.direction, b2.direction, b3.direction
        h1, h2, h3 = b1.high, b2.high, b3.high
        l1, l2, l3 = b1.low, b2.low, b3.low

        # 上升三笔结构：up → down → up
        # 条件：高点抬高 & 回调不破前低
        if d1 == "up" and d2 == "down" and d3 == "up":
            if h3 > h2 and h2 > h1 and l2 > l1:
                long_signals.add(b3.end_index)

        # 下降三笔结构：down → up → down
        # 条件：低点降低 & 反弹不过前高
        if d1 == "down" and d2 == "up" and d3 == "down":
            if l3 < l2 and l2 < l1 and h2 < h1:
                short_signals.add(b3.end_index)

    # --- 单笔大幅趋势延伸信号 ---
    for bi in valid_bis:
        if abs(bi.length_pct) >= 0.02:  # 2% 以上视为趋势延伸
            if bi.direction == "up":
                long_signals.add(bi.end_index)
            else:
                short_signals.add(bi.end_index)

    logging.info(
        f"🧩 结构信号生成完成: long={len(long_signals)}, short={len(short_signals)}"
    )
    return long_signals, short_signals


# ==================================
#  回测核心引擎：run_symbol()
# ==================================

def run_symbol(symbol, df_ltf, df_mtf, df_htf, capital=10000.0):
    """
    symbol: "BTC/USDT"
    df_ltf: 5m
    df_mtf: 1h
    df_htf: 4h
    """

    if df_ltf is None or len(df_ltf) == 0:
        logging.error(f"❌ {symbol} df_ltf 为空，跳过")
        return {
            "symbol": symbol,
            "pnl": 0.0,
            "trades": 0,
            "win_rate": 0.0,
            "struct_score": 0.0,
            "struct_bias": 0.0,
            "ma_score": 0.5,
            "final_score": 0.0,
            "regime": "mixed",
            "bi_total": 0,
            "bi_valid": 0,
            "fractals": 0,
        }

    close = df_ltf["close"].values

    # 1) ATR（止损/止盈使用）
    atr = compute_atr(df_ltf, period=14).fillna(method="bfill").values

    # 2) 缠论基础结构：分型 + 笔 + 有效笔过滤
    fractals = detect_fractals(df_ltf)
    bis = detect_bis(df_ltf, fractals)
    valid_bis = filter_valid_bis(bis, min_bars=7, min_move_pct=0.003)

    # 3) 结构方向因子 + MA 因子
    struct_info = compute_structure_factors(df_ltf, df_mtf, valid_bis)
    struct_score = struct_info["struct_score"]
    struct_bias = struct_info["struct_bias"]
    ma_score = struct_info["ma_score"]
    final_score = struct_info["final_score"]
    regime = struct_info["regime"]

    logging.info(
        f"📐 {symbol} bi_total={len(bis)}, bi_valid={len(valid_bis)}, "
        f"struct={struct_score:.2f}, bias={struct_bias:.2f}, ma={ma_score:.2f}, "
        f"final={final_score:.2f}, regime={regime}"
    )

    # 4) 结构信号
    long_signals, short_signals = generate_structure_signals(df_ltf, valid_bis)

    # ======================================
    #  5) 回测交易执行引擎
    # ======================================
    position = 0                # 0=空仓, 1=多头, -1=空头
    entry_price = 0.0
    pnl = 0.0
    trades = 0
    wins = 0
    cooldown = 0                # 冷静期 bars

    for i in range(len(df_ltf)):

        # 冷静期减少
        if cooldown > 0:
            cooldown -= 1
            continue

        # ===========================
        #  平仓逻辑（止盈 + 止损）
        # ===========================
        if position != 0:
            move = (close[i] - entry_price) / entry_price

            # 动态止盈止损：基于ATR
            sl = -2 * atr[i] / entry_price
            tp = 3 * atr[i] / entry_price

            exit_flag = False

            if position == 1:
                if move <= sl or move >= tp:
                    exit_flag = True
            else:  # 空头
                if -move <= sl or -move >= tp:
                    exit_flag = True

            if exit_flag:
                pnl += move * capital * position
                trades += 1
                if move * position > 0:
                    wins += 1

                position = 0
                cooldown = 86  # 约 3 天 5m K线
                continue

        # ===========================
        #  开仓逻辑：结构信号 + 方向偏置
        # ===========================
        if position == 0:

            # 结构做多信号：需要上行偏置 + 一定强度
            if i in long_signals and struct_bias > 0 and final_score > 0.40:
                position = 1
                entry_price = close[i]
                continue

            # 结构做空信号：需要下行偏置 + 一定强度
            if i in short_signals and struct_bias < 0 and final_score > 0.40:
                position = -1
                entry_price = close[i]
                continue

    # 回测结束后需要关闭持仓
    if position != 0:
        move = (close[-1] - entry_price) / entry_price
        pnl += move * capital * position
        trades += 1
        if move * position > 0:
            wins += 1

    win_rate = wins / trades if trades > 0 else 0.0

    return {
        "symbol": symbol,
        "pnl": round(pnl, 2),
        "trades": trades,
        "win_rate": round(win_rate, 4),
        "struct_score": struct_score,
        "struct_bias": struct_bias,
        "ma_score": ma_score,
        "final_score": final_score,
        "regime": regime,
        "bi_total": len(bis),
        "bi_valid": len(valid_bis),
        "fractals": len(fractals),
    }


# ==================================
#  main 入口
# ==================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--data-source", type=str, default="local")
    args = parser.parse_args()

    logging.info("🚀 SmartBacktest V16_2 启动")

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    results = []
    total_pnl = 0.0

    for sym in symbols:
        logging.info(f"🔍 处理 {sym}")

        # 这里只实现 local 模式，real 模式我们之前在 V11-13 已经有了
        if args.data_source != "local":
            logging.warning(f"⚠️ 当前 V16_2 只实现 data-source=local, 已自动切换为 local")
        df_ltf = load_local_kline(sym, "5m", args.days)
        df_mtf = load_local_kline(sym, "1h", args.days + 3)
        df_htf = load_local_kline(sym, "4h", args.days + 7)

        res = run_symbol(sym, df_ltf, df_mtf, df_htf)
        results.append(res)
        total_pnl += res["pnl"]

    print("\n========== 📈 SmartBacktest V16_2 报告 ==========")
    print(f"总收益: {round(total_pnl,2)}")

    for r in results:
        print(
            f"\n- {r['symbol']}: pnl={r['pnl']}, trades={r['trades']}, win_rate={r['win_rate']}, "
            f"struct={r['struct_score']:.2f}, bias={r['struct_bias']:.2f}, "
            f"ma={r['ma_score']:.2f}, final={r['final_score']:.2f}, regime={r['regime']}, "
            f"Bi_total={r['bi_total']}, Bi_valid={r['bi_valid']}, fractals={r['fractals']}"
        )


if __name__ == "__main__":
    main()
