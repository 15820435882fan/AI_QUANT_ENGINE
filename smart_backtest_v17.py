# smart_backtest_v17.py
# V17_fixed: 本文件自带本地数据加载 + 缠论结构笔 + 结构信号 + 简单回测
# 不依赖 local_data_engine.py，直接从 data/binance/<SYMBOL>/<interval>.csv 读取

import os
import logging
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np

# ===================== 日志配置 =====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("V17")


# ===================== 本地数据加载 =====================

DATA_ROOT = os.path.join("data", "binance")


def _symbol_to_dir(symbol: str) -> str:
    """
    'BTC/USDT' -> 'BTCUSDT'
    """
    return symbol.replace("/", "").upper()


def load_local_kline(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """
    从 data/binance/<SYMBOL>/<interval>.csv 读取本地K线数据，并按最近 days 天截取。

    CSV 要求：
    - 至少包含 timestamp, open, high, low, close 这些列（大小写没关系）
    - timestamp 可以是毫秒整数，也可以是 'YYYY-MM-DD HH:MM:SS' 字符串
    """
    sym_dir = _symbol_to_dir(symbol)
    path = os.path.join(DATA_ROOT, sym_dir, f"{interval}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"本地数据文件不存在: {path}")

    df = pd.read_csv(path)

    # 统一列名为小写
    df.columns = [str(c).lower() for c in df.columns]

    if "timestamp" not in df.columns:
        # 假定第一列是时间
        df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)

    ts_col = df["timestamp"]

    # 判断 timestamp 类型
    if np.issubdtype(ts_col.dtype, np.number):
        # 数字，按毫秒处理
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    else:
        # 字符串时间
        df["timestamp"] = pd.to_datetime(df["timestamp"], infer_datetime_format=True)

    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    # 选择必须列
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"{path} 中缺少列: {col}")

    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(
        float
    )

    end_ts = df.index.max()
    start_ts = end_ts - pd.Timedelta(days=days + 3)  # 多取3天做缓冲
    df = df[df.index >= start_ts]

    logger.info(
        f"📥 [Local V17] 载入本地数据: {symbol} {interval}, 天数≈{days}, 行数={len(df)}"
    )
    return df


# ===================== 技术指标 =====================


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr


# ===================== 缠论结构：分型 & 笔 =====================


@dataclass
class FractalPoint:
    index: int
    price: float
    kind: str  # 'top' or 'bottom'


@dataclass
class BiSegment:
    start_idx: int
    end_idx: int
    direction: str  # 'up' or 'down'
    start_price: float
    end_price: float
    high: float
    low: float


def detect_fractals(df: pd.DataFrame, window: int = 2) -> List[FractalPoint]:
    """
    简单分型识别：
    - 顶分型：当前 high >= 左右 window 根K的 high
    - 底分型：当前 low <= 左右 window 根K的 low
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    fractals: List[FractalPoint] = []
    for i in range(window, n - window):
        left_h = highs[i - window : i].max()
        right_h = highs[i + 1 : i + 1 + window].max()
        left_l = lows[i - window : i].min()
        right_l = lows[i + 1 : i + 1 + window].min()

        if highs[i] >= left_h and highs[i] >= right_h:
            fractals.append(FractalPoint(i, highs[i], "top"))
        elif lows[i] <= left_l and lows[i] <= right_l:
            fractals.append(FractalPoint(i, lows[i], "bottom"))
    return fractals


def build_bis_from_fractals(
    df: pd.DataFrame, frs: List[FractalPoint]
) -> List[BiSegment]:
    """
    简化版笔构建：
    - 分型按 index 排序
    - 连续相同类型分型，只保留更“极端”的一个（top 保留高的，bottom 保留低的）
    - 相邻分型构成一笔，方向取 start -> end 的价格方向
    """
    if not frs:
        return []

    # 1. 按 index 排序
    frs = sorted(frs, key=lambda x: x.index)

    # 2. 合并连续同类分型
    merged: List[FractalPoint] = []
    for fp in frs:
        if not merged:
            merged.append(fp)
        else:
            last = merged[-1]
            if fp.kind == last.kind:
                # 同类分型，top 选更高，bottom 选更低
                if fp.kind == "top":
                    if fp.price >= last.price:
                        merged[-1] = fp
                else:  # bottom
                    if fp.price <= last.price:
                        merged[-1] = fp
            else:
                merged.append(fp)

    if len(merged) < 2:
        return []

    close = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    bis: List[BiSegment] = []

    for i in range(len(merged) - 1):
        s = merged[i]
        e = merged[i + 1]
        start_idx = s.index
        end_idx = e.index
        if end_idx <= start_idx:
            continue
        start_price = close[start_idx]
        end_price = close[end_idx]
        direction = "up" if end_price >= start_price else "down"
        seg_high = highs[start_idx : end_idx + 1].max()
        seg_low = lows[start_idx : end_idx + 1].min()
        bis.append(
            BiSegment(
                start_idx=start_idx,
                end_idx=end_idx,
                direction=direction,
                start_price=start_price,
                end_price=end_price,
                high=float(seg_high),
                low=float(seg_low),
            )
        )

    return bis


# ===================== 中枢粗略识别 =====================


def detect_zhongshu(bis: List[BiSegment]) -> int:
    """
    非严格缠论中枢，仅用于统计强震荡结构数量：
    - 三笔一组，方向交替（up-down-up 或 down-up-down）
    - 三笔价格区间有重叠，则认为存在一个中枢
    """
    if len(bis) < 3:
        return 0

    zcount = 0
    for i in range(len(bis) - 2):
        b1, b2, b3 = bis[i], bis[i + 1], bis[i + 2]
        if not (
            b1.direction != b2.direction
            and b2.direction != b3.direction
            and b1.direction == b3.direction
        ):
            continue

        # 区间交集
        hi1, lo1 = b1.high, b1.low
        hi2, lo2 = b2.high, b2.low
        hi3, lo3 = b3.high, b3.low

        top = min(hi1, hi2, hi3)
        bottom = max(lo1, lo2, lo3)
        if top > bottom:
            zcount += 1

    return zcount


# ===================== 趋势评分（基于笔 + 中枢） =====================


def compute_trend_score(bis: List[BiSegment], zhongshu_count: int) -> Dict[str, float]:
    total = len(bis)
    if total == 0:
        return {"trend_up": 0.0, "trend_down": 0.0, "range": 0.0}

    ups = sum(1 for b in bis if b.direction == "up")
    downs = sum(1 for b in bis if b.direction == "down")

    trend_up = ups / total
    trend_down = downs / total
    # 用中枢数量粗略表示震荡强度
    range_factor = min(zhongshu_count / max(total - 2, 1), 1.0)

    return {
        "trend_up": float(trend_up),
        "trend_down": float(trend_down),
        "range": float(range_factor),
    }


# ===================== 结构信号生成（笔结构 + RR 约束） =====================


@dataclass
class StructureSignal:
    index: int  # 对应 5m K线的行号（iloc）
    direction: str  # 'long' or 'short'
    entry: float
    sl: float
    tp: float
    strength: float  # 0~1 大致代表结构强度


def generate_structure_signals(
    df: pd.DataFrame,
    bis: List[BiSegment],
    trend_info: Dict[str, float],
    min_bars: int = 5,
    min_move_pct: float = 0.002,
    rr: float = 2.0,
) -> List[StructureSignal]:
    """
    简化版结构信号：
    - 使用三笔结构：up-down-up 认为是向上结构；down-up-down 为向下结构
    - 要求三笔的高低点具备“创新高/新低”特征
    - 停损放在中间那笔的极端价稍外一点
    - 目标价按 RR=2 放大
    - 若全局趋势明显，则只取趋势同向信号
    """
    if len(bis) < 3:
        return []

    close = df["close"].values
    signals: List[StructureSignal] = []

    trend_up = trend_info.get("trend_up", 0.0)
    trend_down = trend_info.get("trend_down", 0.0)

    # 简单定义“主趋势方向”
    if trend_up - trend_down > 0.15:
        major = "up"
    elif trend_down - trend_up > 0.15:
        major = "down"
    else:
        major = "mixed"

    for i in range(len(bis) - 2):
        b1, b2, b3 = bis[i], bis[i + 1], bis[i + 2]

        # 三笔方向要交替，且 1 和 3 同向
        if not (
            b1.direction != b2.direction
            and b2.direction != b3.direction
            and b1.direction == b3.direction
        ):
            continue

        bars_len = b3.end_idx - b1.start_idx + 1
        if bars_len < min_bars:
            continue

        # 上涨结构：up - down - up + 创新高 + 高低抬升
        if b1.direction == "up":
            # 创新高：第3笔高点 > 第1笔高点
            if not (b3.high > b1.high and b3.low >= b1.low):
                continue

            entry_idx = b3.end_idx
            if entry_idx >= len(close):
                continue
            entry_price = float(close[entry_idx])

            sl_price = min(b1.low, b2.low, b3.low) * 0.998  # 稍微再放一点
            if sl_price >= entry_price:
                continue

            move_pct = (entry_price - sl_price) / entry_price
            if move_pct < min_move_pct:
                continue

            tp_price = entry_price + (entry_price - sl_price) * rr

            # 趋势过滤：如果大趋势明显向下，就少做多
            if major == "down":
                continue

            strength = float(trend_up - trend_down + 0.5)  # 大致落在 0~1

            signals.append(
                StructureSignal(
                    index=entry_idx,
                    direction="long",
                    entry=entry_price,
                    sl=float(sl_price),
                    tp=float(tp_price),
                    strength=max(0.0, min(1.0, strength)),
                )
            )

        # 下跌结构：down - up - down + 新低 + 高低降低
        else:
            if not (b3.low < b1.low and b3.high <= b1.high):
                continue

            entry_idx = b3.end_idx
            if entry_idx >= len(close):
                continue
            entry_price = float(close[entry_idx])

            sl_price = max(b1.high, b2.high, b3.high) * 1.002
            if sl_price <= entry_price:
                continue

            move_pct = (sl_price - entry_price) / entry_price
            if move_pct < min_move_pct:
                continue

            tp_price = entry_price - (sl_price - entry_price) * rr

            if major == "up":
                continue

            strength = float(trend_down - trend_up + 0.5)

            signals.append(
                StructureSignal(
                    index=entry_idx,
                    direction="short",
                    entry=entry_price,
                    sl=float(sl_price),
                    tp=float(tp_price),
                    strength=max(0.0, min(1.0, strength)),
                )
            )

    # 去重：同一个 index 可能出现多个信号，保留强度最大的一个
    by_index: Dict[int, StructureSignal] = {}
    for sig in signals:
        old = by_index.get(sig.index)
        if old is None or sig.strength > old.strength:
            by_index[sig.index] = sig

    final_signals = sorted(by_index.values(), key=lambda s: s.index)
    logger.info(f"🧩 结构信号生成完成: count={len(final_signals)}")
    return final_signals


# ===================== 简单回测引擎 =====================


@dataclass
class TradeRecord:
    entry_idx: int
    exit_idx: int
    direction: str
    entry_price: float
    exit_price: float
    pnl: float


def backtest_with_signals(
    df: pd.DataFrame,
    signals: List[StructureSignal],
    capital: float = 10_000.0,
) -> Tuple[float, List[TradeRecord], float]:
    """
    非杠杆单币种回测：
    - 每次信号全仓建一个方向（long/short），不叠加仓位
    - 止盈/止损 或 反向突破触发平仓
    - 没有持仓时才会响应新信号
    """
    if df.empty:
        return 0.0, [], 0.0

    close = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    # 按 index 建立查找表
    sig_map: Dict[int, StructureSignal] = {s.index: s for s in signals}

    equity = capital
    peak_equity = capital
    max_dd = 0.0  # 负数代表回撤比例

    position: Optional[StructureSignal] = None
    pos_size: float = 0.0  # 持仓数量（币的数量）
    trades: List[TradeRecord] = []

    n = len(df)

    for i in range(n):
        price = float(close[i])

        # 有仓位：检查止损/止盈
        if position is not None:
            if position.direction == "long":
                # 先检查止损
                if lows[i] <= position.sl:
                    exit_price = position.sl
                # 再检查止盈
                elif highs[i] >= position.tp:
                    exit_price = position.tp
                else:
                    exit_price = None
            else:  # short
                if highs[i] >= position.sl:
                    exit_price = position.sl
                elif lows[i] <= position.tp:
                    exit_price = position.tp
                else:
                    exit_price = None

            if exit_price is not None:
                pnl = (exit_price - position.entry) * pos_size
                equity += pnl
                trades.append(
                    TradeRecord(
                        entry_idx=position.index,
                        exit_idx=i,
                        direction=position.direction,
                        entry_price=position.entry,
                        exit_price=float(exit_price),
                        pnl=float(pnl),
                    )
                )
                # 回撤
                if equity > peak_equity:
                    peak_equity = equity
                dd = equity / peak_equity - 1.0
                if dd < max_dd:
                    max_dd = dd

                position = None
                pos_size = 0.0

        # 没仓位：检查是否有信号
        if position is None:
            sig = sig_map.get(i)
            if sig is not None:
                # 以当前 equity 计算仓位
                if sig.entry <= 0:
                    continue
                pos_size = equity / sig.entry
                position = sig

    # 收尾：若还有仓位，最后一根bar收盘价平仓
    if position is not None:
        final_price = float(close[-1])
        if position.direction == "long":
            pnl = (final_price - position.entry) * pos_size
        else:
            pnl = (position.entry - final_price) * pos_size
        equity += pnl
        trades.append(
            TradeRecord(
                entry_idx=position.index,
                exit_idx=n - 1,
                direction=position.direction,
                entry_price=position.entry,
                exit_price=final_price,
                pnl=float(pnl),
            )
        )
        if equity > peak_equity:
            peak_equity = equity
        dd = equity / peak_equity - 1.0
        if dd < max_dd:
            max_dd = dd

    total_pnl = equity - capital
    return float(total_pnl), trades, float(max_dd)


# ===================== 单币种执行逻辑 =====================


def run_symbol(
    symbol: str, days: int, data_source: str, capital: float
) -> Dict[str, object]:
    """
    核心执行流程：
    1. 载入 5m / 1h / 4h 数据（目前主要用 5m 做结构，1h/4h 预留）
    2. 检测分型 + 笔
    3. 统计中枢数量，计算趋势评分
    4. 生成结构信号
    5. 简单回测
    """
    if data_source != "local":
        logger.warning("当前 V17_fixed 仅支持 data_source=local，已自动使用 local。")

    df_ltf = load_local_kline(symbol, "5m", days)
    df_mtf = load_local_kline(symbol, "1h", days + 3)
    df_htf = load_local_kline(symbol, "4h", days + 7)

    logger.info(
        f"📥 {symbol} 5m={len(df_ltf)}, 1h={len(df_mtf)}, 4h={len(df_htf)} (source=local)"
    )

    # 分型 & 笔（用 5m）
    frs = detect_fractals(df_ltf, window=2)
    bis = build_bis_from_fractals(df_ltf, frs)
    zss = detect_zhongshu(bis)
    trend_info = compute_trend_score(bis, zss)

    logger.info(
        f"📐 {symbol} trend_up={trend_info['trend_up']:.2f}, "
        f"trend_down={trend_info['trend_down']:.2f}, range={trend_info['range']:.2f}, "
        f"bis={len(bis)}, zss={zss}"
    )

    signals = generate_structure_signals(
        df_ltf,
        bis,
        trend_info,
        min_bars=5,
        min_move_pct=0.0015,  # 放宽一点，信号会多一些
        rr=2.0,
    )
    logger.info(f"🧩 {symbol} 生成结构信号: {len(signals)}")

    pnl, trades, max_dd = backtest_with_signals(df_ltf, signals, capital=capital)

    wins = sum(1 for t in trades if t.pnl > 0)
    total_trades = len(trades)
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    return {
        "symbol": symbol,
        "pnl": pnl,
        "trades": total_trades,
        "win_rate": win_rate,
        "max_dd": max_dd,
        "trend": trend_info,
        "bis": len(bis),
        "zss": zss,
        "signals": len(signals),
    }


# ===================== 主程序 =====================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmartBacktest V17_fixed")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC/USDT,ETH/USDT",
        help="逗号分隔的交易对，例如: BTC/USDT,ETH/USDT",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="回测天数（使用最近 N 天的本地数据）",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="local",
        help="数据源（当前仅支持 local）",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10_000.0,
        help="每个币种分配的初始资金",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]

    logger.info("🚀 SmartBacktest V17_fixed 启动")
    logger.info(f"🪙 币种: {syms}")
    logger.info(f"📅 回测天数: {args.days}")
    logger.info(f"📊 数据源: {args.data_source}")

    all_results: List[Dict[str, object]] = []

    for sym in syms:
        try:
            res = run_symbol(sym, args.days, args.data_source, args.capital)
            all_results.append(res)
        except Exception as e:
            logger.exception(f"❌ {sym} 处理失败: {e}")

    # 汇总结果
    total_pnl = sum(r["pnl"] for r in all_results)
    total_trades = sum(r["trades"] for r in all_results)
    total_wins = sum(int(r["trades"] * r["win_rate"]) for r in all_results)
    win_rate = total_wins / total_trades if total_trades > 0 else 0.0
    max_dd = min((r["max_dd"] for r in all_results), default=0.0)

    print("\n========== 📈 SmartBacktest V17_fixed 报告 ==========")
    print(f"总收益: {total_pnl:.2f}")
    print(f"总交易数: {total_trades}")
    print(f"总胜率: {win_rate*100:.2f}%")
    print(f"最大回撤: {max_dd*100:.2f}%")

    print("\n按币种：")
    for r in all_results:
        trend = r["trend"]
        print(
            f"- {r['symbol']}: pnl={r['pnl']:.2f}, trades={r['trades']}, "
            f"win_rate={r['win_rate']*100:.2f}%, maxDD={r['max_dd']*100:.2f}%, "
            f"trend_up={trend['trend_up']:.2f}, trend_down={trend['trend_down']:.2f}, "
            f"range={trend['range']:.2f}, bis={r['bis']}, zss={r['zss']}, signals={r['signals']}"
        )


if __name__ == "__main__":
    main()
