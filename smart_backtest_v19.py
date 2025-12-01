"""
smart_backtest_v19.py

V19: 多级别缠论 + AI 评分 + 结构化风控 回测引擎（雏形可跑版）

说明：
- 依赖 local_data_engine.load_local_kline() 读取本地 K 线
- 默认使用多周期：4h / 1h / 15m / 5m
- 本文件目标：给出一个「能跑通 + 结构清晰」的 V19 主干框架
  方便后续逐步替换/增强具体算法细节
"""

import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    # 与第一季保持兼容
    from local_data_engine import load_local_kline
except Exception as e:  # pragma: no cover - 仅防御性处理
    load_local_kline = None  # type: ignore

    def _missing_loader(*args, **kwargs):
        raise RuntimeError(
            "未找到 local_data_engine.load_local_kline，"
            "请确认 local_data_engine.py 在同一目录，"
            "并且包含函数 load_local_kline(symbol, interval, days)"
        )

    load_local_kline = _missing_loader  # type: ignore


# ===================== 日志配置 =====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ===================== 基本数据结构 =====================


@dataclass
class Bi:
    start_index: int
    end_index: int
    direction: str  # "up" / "down"


@dataclass
class ZhongShu:
    start_index: int
    end_index: int
    high: float
    low: float


@dataclass
class ChanSignal:
    index: int               # 在执行级别(5m)中的索引
    price: float
    kind: str                # third_buy / third_sell / breakout_buy / breakout_sell 等
    direction: str           # long / short
    score: float             # AI 评分 0~1
    rr: float                # 期望收益风险比
    sl: float                # 止损价
    tp: float                # 止盈价


@dataclass
class Trade:
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    direction: str           # long / short

    @property
    def pnl(self) -> float:
        if self.direction == "long":
            return self.exit_price - self.entry_price
        else:
            return self.entry_price - self.exit_price


# ===================== 工具函数 =====================


def ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """确保 DataFrame 拥有 OHLC 列并按时间排序、设置 DatetimeIndex。"""
    if "timestamp" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="ignore")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp").set_index("timestamp")

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    # 有些数据 index 已经是 DatetimeIndex，这里做一次兜底
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
        df.index = pd.to_datetime(df.index)

    return df[["open", "high", "low", "close"]].copy()


# ===================== 单级别缠论结构（简化可跑版） =====================


def detect_fractals(df: pd.DataFrame) -> Tuple[List[int], List[int]]:
    """
    简化版分型检测：
    - 顶分型：high[i] > high[i-1] & high[i] > high[i+1]
    - 底分型：low[i]  < low[i-1]  & low[i]  < low[i+1]
    """
    highs = df["high"].values
    lows = df["low"].values
    up_f: List[int] = []
    down_f: List[int] = []

    for i in range(2, len(df) - 2):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            up_f.append(i)
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            down_f.append(i)

    return up_f, down_f


def detect_bis(
    df: pd.DataFrame,
    up_f: List[int],
    down_f: List[int],
    min_bi_height: float = 0.002,
) -> List[Bi]:
    """
    简化版“笔”识别逻辑：
    - 使用顶/底分型按时间排序
    - 相邻两个分型若方向不同、且价格差超过阈值则视为一笔
    """
    highs = df["high"].values
    lows = df["low"].values

    all_f = sorted(set(up_f + down_f))
    if len(all_f) < 2:
        return []

    bis: List[Bi] = []

    def is_top(idx: int) -> bool:
        return idx in up_f

    def is_bottom(idx: int) -> bool:
        return idx in down_f

    last_idx = all_f[0]
    last_dir: Optional[str] = None

    for idx in all_f[1:]:
        if is_top(last_idx) and is_bottom(idx):
            # down 笔
            high_price = highs[last_idx]
            low_price = lows[idx]
            if (high_price - low_price) / high_price >= min_bi_height:
                if last_dir != "down":
                    bis.append(Bi(start_index=last_idx, end_index=idx, direction="down"))
                    last_dir = "down"
                    last_idx = idx
        elif is_bottom(last_idx) and is_top(idx):
            # up 笔
            low_price = lows[last_idx]
            high_price = highs[idx]
            if (high_price - low_price) / low_price >= min_bi_height:
                if last_dir != "up":
                    bis.append(Bi(start_index=last_idx, end_index=idx, direction="up"))
                    last_dir = "up"
                    last_idx = idx
        else:
            # 同向分型，择优保留极值
            if is_top(last_idx) and is_top(idx):
                if highs[idx] > highs[last_idx]:
                    last_idx = idx
            if is_bottom(last_idx) and is_bottom(idx):
                if lows[idx] < lows[last_idx]:
                    last_idx = idx

    return bis


def detect_zhongshu(df: pd.DataFrame, bis: List[Bi]) -> List[ZhongShu]:
    """
    极简版中枢识别：
    - 连续三笔的价格重叠区间视为一个中枢
    - 这里只给出一个近似可跑版本，后续可以替换为你更严谨的缠论实现
    """
    zs_list: List[ZhongShu] = []
    highs = df["high"].values
    lows = df["low"].values

    if len(bis) < 3:
        return zs_list

    for i in range(len(bis) - 2):
        b1, b2, b3 = bis[i], bis[i + 1], bis[i + 2]

        high1 = max(highs[b1.start_index:b1.end_index + 1])
        low1 = min(lows[b1.start_index:b1.end_index + 1])
        high2 = max(highs[b2.start_index:b2.end_index + 1])
        low2 = min(lows[b2.start_index:b2.end_index + 1])
        high3 = max(highs[b3.start_index:b3.end_index + 1])
        low3 = min(lows[b3.start_index:b3.end_index + 1])

        upper = min(high1, high2, high3)
        lower = max(low1, low2, low3)

        if upper > lower:
            start_index = min(b1.start_index, b2.start_index, b3.start_index)
            end_index = max(b1.end_index, b2.end_index, b3.end_index)
            zs_list.append(ZhongShu(start_index=start_index, end_index=end_index, high=upper, low=lower))

    return zs_list


# ===================== 多级别结构上下文 =====================


@dataclass
class MultiLevelContext:
    symbol: str
    interval_dfs: Dict[str, pd.DataFrame]       # "4h" / "1h" / "15m" / "5m"
    bis: Dict[str, List[Bi]]
    zhongshus: Dict[str, List[ZhongShu]]


def build_multilevel_context(
    symbol: str,
    intervals: List[str],
    days: int,
) -> MultiLevelContext:
    """加载多周期数据并构建基础缠论结构。"""
    interval_dfs: Dict[str, pd.DataFrame] = {}
    bis_map: Dict[str, List[Bi]] = {}
    zs_map: Dict[str, List[ZhongShu]] = {}

    for interval in intervals:
        logging.info(f"[{symbol}] 加载本地数据: {interval}, 最近 {days} 天")
        df = load_local_kline(symbol, interval, days)
        df = ensure_ohlc(df)
        up_f, down_f = detect_fractals(df)
        bis = detect_bis(df, up_f, down_f)
        zhongshus = detect_zhongshu(df, bis)

        interval_dfs[interval] = df
        bis_map[interval] = bis
        zs_map[interval] = zhongshus

        logging.info(
            f"[{symbol}][{interval}] 分型: {len(up_f)+len(down_f)}, 笔: {len(bis)}, 中枢: {len(zhongshus)}"
        )

    return MultiLevelContext(symbol=symbol, interval_dfs=interval_dfs, bis=bis_map, zhongshus=zs_map)


# ===================== AI 评分模型（简化版） =====================


def compute_trend_strength(df: pd.DataFrame, window_short: int = 20, window_long: int = 60) -> float:
    """
    简化趋势强度：
    - 短期均线与长期均线的差异 & 斜率
    - 返回 [-1, 1]，>0 为上升趋势，<0 为下降趋势
    """
    closes = df["close"].values
    if len(closes) < window_long + 2:
        return 0.0

    short_ma = pd.Series(closes).rolling(window_short).mean().iloc[-1]
    long_ma = pd.Series(closes).rolling(window_long).mean().iloc[-1]

    # 斜率（近几根短期均线变化）
    short_series = pd.Series(closes).rolling(window_short).mean().dropna()
    if len(short_series) < 5:
        slope = 0.0
    else:
        slope = short_series.iloc[-1] - short_series.iloc[-5]

    raw = 0.0
    if long_ma != 0 and not np.isnan(long_ma):
        raw = (short_ma - long_ma) / abs(long_ma)

    raw += slope / (abs(long_ma) + 1e-9)

    return float(max(-1.0, min(1.0, raw)))


def ai_score_signal(
    ctx: MultiLevelContext,
    base_interval: str,
    signal_index: int,
    direction: str,
) -> float:
    """
    多级别 AI 评分（雏形）：
    - 4h / 1h / 15m 趋势共振
    - 分数范围 [0,1]
    """
    weights = {"4h": 0.4, "1h": 0.3, "15m": 0.3}
    score = 0.0

    for interval, w in weights.items():
        df = ctx.interval_dfs.get(interval)
        if df is None or len(df) < 100:
            continue

        trend = compute_trend_strength(df)
        if direction == "long":
            contrib = max(0.0, trend) * w
        else:
            contrib = max(0.0, -trend) * w

        score += contrib

    # 归一到 [0,1]
    return float(max(0.0, min(1.0, score)))


# ===================== 风控模块（RR≥2 雏形） =====================


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if not np.isnan(atr) else float(tr.mean())


def compute_risk_for_signal(
    df_exec: pd.DataFrame,
    idx: int,
    direction: str,
    rr_target: float = 2.0,
) -> Tuple[float, float, float]:
    """
    基于 ATR 的简化结构止损/止盈：
    - 止损: entry ± 1 ATR
    - 止盈: entry ± rr_target * ATR
    """
    closes = df_exec["close"].values
    entry_price = float(closes[idx])
    atr = compute_atr(df_exec)

    if atr <= 0 or np.isnan(atr):
        atr = entry_price * 0.002  # fallback 0.2%

    if direction == "long":
        sl = entry_price - atr
        tp = entry_price + rr_target * atr
    else:
        sl = entry_price + atr
        tp = entry_price - rr_target * atr

    rr = abs(tp - entry_price) / max(1e-9, abs(entry_price - sl))
    return sl, tp, rr


# ===================== 信号生成器（多级别 + 执行级别） =====================


def generate_signals_v19(
    ctx: MultiLevelContext,
    exec_interval: str = "5m",
    rr_target: float = 2.0,
    min_score: float = 0.6,
) -> List[ChanSignal]:
    """
    非完美缠论版，但可跑、可迭代：
    - 在执行级别上使用简化趋势反转逻辑
    - 结合多级别 AI 评分进行过滤
    """
    df_exec = ctx.interval_dfs[exec_interval]
    closes = df_exec["close"].values

    signals: List[ChanSignal] = []

    # 简化趋势反转：短均线突破长均线
    short_win = 10
    long_win = 30
    close_s = pd.Series(closes)
    short_ma = close_s.rolling(short_win).mean()
    long_ma = close_s.rolling(long_win).mean()

    last_dir: Optional[str] = None

    for i in range(long_win + 5, len(df_exec) - 5):
        if np.isnan(short_ma.iloc[i]) or np.isnan(long_ma.iloc[i]):
            continue

        dir_now: Optional[str] = None
        if short_ma.iloc[i] > long_ma.iloc[i] and short_ma.iloc[i - 1] <= long_ma.iloc[i - 1]:
            dir_now = "long"
        elif short_ma.iloc[i] < long_ma.iloc[i] and short_ma.iloc[i - 1] >= long_ma.iloc[i - 1]:
            dir_now = "short"

        if dir_now is None or dir_now == last_dir:
            continue

        # 多级别 AI 评分
        score = ai_score_signal(ctx, exec_interval, i, dir_now)
        if score < min_score:
            continue

        # 风控（RR≥2）
        sl, tp, rr = compute_risk_for_signal(df_exec, i, dir_now, rr_target=rr_target)
        if rr < 1.8:  # 稍微放宽一点
            continue

        kind = "trend_break_buy" if dir_now == "long" else "trend_break_sell"
        sig = ChanSignal(
            index=i,
            price=float(closes[i]),
            kind=kind,
            direction=dir_now,
            score=score,
            rr=rr,
            sl=sl,
            tp=tp,
        )
        signals.append(sig)
        last_dir = dir_now

    logging.info(f"[{ctx.symbol}] 生成信号数量: {len(signals)}")
    return signals


# ===================== 回测主逻辑 =====================


def backtest_signals(
    df_exec: pd.DataFrame,
    signals: List[ChanSignal],
    max_hold_bars: int = 200,
) -> Tuple[List[Trade], float]:
    """
    简化版回测：
    - 同一时间仅持有一笔仓位
    - 触达止损/止盈 或 超过最大持仓 bar 数则平仓
    """
    trades: List[Trade] = []
    if not signals:
        return trades, 0.0

    closes = df_exec["close"].values
    highs = df_exec["high"].values
    lows = df_exec["low"].values

    current_pos: Optional[Trade] = None

    for sig in signals:
        if current_pos is not None:
            # 已有持仓，暂时不并行开多仓，简单版本直接忽略后续信号
            continue

        entry_idx = sig.index
        entry_price = sig.price
        direction = sig.direction

        exit_price: Optional[float] = None
        exit_idx: int = entry_idx

        for j in range(entry_idx + 1, min(entry_idx + max_hold_bars, len(df_exec))):
            high = highs[j]
            low = lows[j]

            # 多头：先看止损，再看止盈（保守）
            if direction == "long":
                if low <= sig.sl:
                    exit_price = sig.sl
                    exit_idx = j
                    break
                if high >= sig.tp:
                    exit_price = sig.tp
                    exit_idx = j
                    break
            else:
                if high >= sig.sl:
                    exit_price = sig.sl
                    exit_idx = j
                    break
                if low <= sig.tp:
                    exit_price = sig.tp
                    exit_idx = j
                    break

        if exit_price is None:
            # 未触发止盈/止损，用最后一个可见 close 平仓
            exit_idx = min(entry_idx + max_hold_bars, len(df_exec) - 1)
            exit_price = float(closes[exit_idx])

        trade = Trade(
            entry_index=entry_idx,
            exit_index=exit_idx,
            entry_price=entry_price,
            exit_price=exit_price,
            direction=direction,
        )
        trades.append(trade)

    total_pnl = sum(t.pnl for t in trades)
    return trades, total_pnl


def run_symbol_v19(
    symbol: str,
    days: int,
    intervals: Optional[List[str]] = None,
) -> Dict[str, float]:
    if intervals is None:
        intervals = ["4h", "1h", "15m", "5m"]

    logging.info(f"========== 开始回测 V19: {symbol} ==========")
    ctx = build_multilevel_context(symbol, intervals, days)
    df_exec = ctx.interval_dfs["5m"]

    signals = generate_signals_v19(ctx, exec_interval="5m")
    trades, total_pnl = backtest_signals(df_exec, signals)

    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl <= 0)

    logging.info(f"[{symbol}] 交易笔数: {len(trades)}, 胜率: {wins / max(1, len(trades)):.2f}, 总收益: {total_pnl:.4f}")

    return {
        "symbol": symbol,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "pnl": total_pnl,
    }


# ===================== CLI 入口 =====================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SmartBacktest V19 - 多级别缠论 + AI 评分 回测引擎（雏形版）"
    )
    p.add_argument(
        "--symbols",
        type=str,
        required=True,
        help="回测币种列表，例如: BTCUSDT,ETHUSDT,BNBUSDT",
    )
    p.add_argument(
        "--days",
        type=int,
        default=90,
        help="回测区间天数（对所有周期统一使用）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]

    total_pnl = 0.0
    total_trades = 0
    total_wins = 0

    for sym in syms:
        try:
            res = run_symbol_v19(sym, args.days)
        except Exception as e:
            logging.exception(f"[{sym}] 回测失败: {e}")
            continue

        total_pnl += res["pnl"]
        total_trades += res["trades"]
        total_wins += res["wins"]

    print("\n========== 📈 V19 多级别缠论 AI 回测战报 ==========")
    print(f"💰 总收益: {total_pnl:.4f}")
    print(f"🔢 总交易数: {total_trades}")
    if total_trades > 0:
        print(f"🎯 综合胜率: {total_wins / total_trades:.2f}")
    else:
        print("🎯 综合胜率: N/A (无交易)")


if __name__ == "__main__":
    main()
