"""
smart_backtest_v19_2.py

V19_2: 多级别缠论结构触发 + AI 评分 + 结构化风控 回测引擎

与 V19_1 的核心区别：
1）彻底移除 MA10/MA30 交叉触发逻辑
2）信号基于「缠论结构触发」：
    - 笔高/低点突破
    - 中枢上下沿突破
3）引入结构稳定度评分 + 多级别趋势评分 → 综合 AI score ∈ [0,1]
4）保留 ATR 风控与 RR 计算，但阈值较宽，让策略先“活起来”再精炼
"""

import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from local_data_engine import load_local_kline
except Exception:
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
    kind: str                # bi_break / zs_break_up / zs_break_down 等
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
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="ignore")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp").set_index("timestamp")

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

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
    min_bi_height: float = 0.0015,
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
    - 这里只给出一个近似可跑版本，后续可以替换为更严谨实现
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


# ===================== AI 评分模型 =====================


def compute_trend_strength(df: pd.DataFrame, window_short: int = 20, window_long: int = 60) -> float:
    """
    简化趋势强度：
    - 短期均线与长期均线的差异 & 斜率
    - 返回 [-1, 1]，>0 为上升趋势，<0 为下降趋势
    """
    closes = df["close"].values
    if len(closes) < window_long + 5:
        return 0.0

    series = pd.Series(closes)
    short_ma_series = series.rolling(window_short).mean()
    long_ma_series = series.rolling(window_long).mean()

    short_ma = short_ma_series.iloc[-1]
    long_ma = long_ma_series.iloc[-1]

    if pd.isna(short_ma) or pd.isna(long_ma):
        return 0.0

    valid_short = short_ma_series.dropna()
    if len(valid_short) < 5:
        slope = 0.0
    else:
        slope = valid_short.iloc[-1] - valid_short.iloc[-5]

    raw = 0.0
    if long_ma != 0 and not np.isnan(long_ma):
        raw = (short_ma - long_ma) / abs(long_ma)

    raw += slope / (abs(long_ma) + 1e-9)

    return float(max(-1.0, min(1.0, raw)))


def structural_stability_score(
    df_exec: pd.DataFrame,
    bis_exec: List[Bi],
    zhongshus_exec: List[ZhongShu],
    idx: int,
) -> float:
    """
    结构稳定度评分（0~1），考虑：
    - 最近一笔的长度（时间跨度）
    - 最近中枢是否存在、且 idx 附近在中枢外（趋势更清晰）
    - 价格相对于中枢上下沿的位置
    """
    if len(bis_exec) == 0:
        return 0.1

    closes = df_exec["close"].values

    # 最近一笔
    last_bi = bis_exec[-1]
    bi_len = last_bi.end_index - last_bi.start_index
    bi_len_score = min(1.0, bi_len / 50.0)  # 50 根以上按满分算

    zs_score = 0.2
    pos_score = 0.2

    if len(zhongshus_exec) > 0:
        last_zs = zhongshus_exec[-1]
        # 中枢的有效宽度
        zs_width = last_zs.high - last_zs.low
        if zs_width <= 0:
            zs_width = closes[last_zs.end_index] * 0.001

        zs_score = 0.3  # 有中枢基础加分

        price = closes[idx]
        # 如果价格已经偏离中枢区间，说明趋势已经走出，方向更明确
        if price > last_zs.high:
            pos_score = 0.5
        elif price < last_zs.low:
            pos_score = 0.5
        else:
            # 在中枢内部，趋势不明确
            pos_score = 0.1

    total = 0.3 * bi_len_score + 0.4 * zs_score + 0.3 * pos_score
    return float(max(0.0, min(1.0, total)))


def ai_score_signal(
    ctx: MultiLevelContext,
    signal_index: int,
    direction: str,
) -> float:
    """
    综合 AI 评分：
    - 4h / 1h / 15m 趋势强度
    - 5m 结构稳定度
    """
    weights_trend = {"4h": 0.4, "1h": 0.3, "15m": 0.3}
    trend_score = 0.0

    for interval, w in weights_trend.items():
        df = ctx.interval_dfs.get(interval)
        if df is None or len(df) < 100:
            continue

        t = compute_trend_strength(df)
        if direction == "long":
            contrib = max(0.0, t) * w
        else:
            contrib = max(0.0, -t) * w
        trend_score += contrib

    # 结构稳定度（来自 5m）
    df_exec = ctx.interval_dfs["5m"]
    bis_exec = ctx.bis["5m"]
    zs_exec = ctx.zhongshus["5m"]
    struct_score = structural_stability_score(df_exec, bis_exec, zs_exec, signal_index)

    # 线性融合并限制在 [0,1]
    score = 0.6 * trend_score + 0.4 * struct_score
    return float(max(0.0, min(1.0, score)))


# ===================== 风控模块 =====================


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
    rr_target: float = 1.5,
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


# ===================== 结构触发信号生成器 =====================


def generate_signals_v19_2(
    ctx: MultiLevelContext,
    exec_interval: str = "5m",
    rr_target: float = 1.5,
    min_score: float = 0.35,
    bi_break_gap: float = 0.0005,
    zs_break_gap: float = 0.0005,
) -> List[ChanSignal]:
    """
    V19_2 信号逻辑：
    1）笔突破触发：
        - 向上笔：当前价格突破该笔最高价 * (1 + bi_break_gap) → long
        - 向下笔：当前价格跌破该笔最低价 * (1 - bi_break_gap) → short
    2）中枢突破触发：
        - 当前价格突破最近中枢 high * (1 + zs_break_gap) → long
        - 跌破最近中枢 low * (1 - zs_break_gap) → short
    3）AI 评分过滤：
        - score = 0.6 * 多级别趋势 + 0.4 * 结构稳定度
        - 仅当 score ≥ min_score 才执行
    4）RR 过滤：
        - 仅保留 RR ≥ 1.2 的信号
    """
    df_exec = ctx.interval_dfs[exec_interval]
    bis_exec = ctx.bis[exec_interval]
    zs_exec = ctx.zhongshus[exec_interval]
    closes = df_exec["close"].values
    highs = df_exec["high"].values
    lows = df_exec["low"].values

    signals: List[ChanSignal] = []

    if len(df_exec) < 100:
        return signals

    # 为了避免太前面的结构，使用最近 N 根 K 线范围内的笔 & 中枢
    max_lookback_bars = 600  # 大约 2 天的 5m 数据
    last_index = len(df_exec) - 1
    start_bar = max(0, last_index - max_lookback_bars)

    # 预先筛选：只看 end_index 在区间内的笔
    candidate_bis = [b for b in bis_exec if b.end_index >= start_bar]
    candidate_zs = [z for z in zs_exec if z.end_index >= start_bar]

    for i in range(start_bar + 5, len(df_exec) - 5):
        price = closes[i]
        triggered = False
        dir_now: Optional[str] = None
        kind = ""

        # ---------- 1）笔突破 ----------
        # 找到距离 i 最近的一笔
        nearest_bi: Optional[Bi] = None
        min_dist = 10**9
        for b in candidate_bis:
            if b.end_index <= i and i - b.end_index < min_dist:
                nearest_bi = b
                min_dist = i - b.end_index

        if nearest_bi is not None:
            bi_high = max(highs[nearest_bi.start_index:nearest_bi.end_index + 1])
            bi_low = min(lows[nearest_bi.start_index:nearest_bi.end_index + 1])

            if nearest_bi.direction == "up":
                # 上升笔高点突破
                if price > bi_high * (1 + bi_break_gap):
                    dir_now = "long"
                    kind = "bi_break_up"
                    triggered = True
            elif nearest_bi.direction == "down":
                # 下降笔低点突破
                if price < bi_low * (1 - bi_break_gap):
                    dir_now = "short"
                    kind = "bi_break_down"
                    triggered = True

        # ---------- 2）中枢突破 ----------
        if not triggered and len(candidate_zs) > 0:
            # 取最近一个中枢
            last_zs = candidate_zs[-1]
            zs_high = last_zs.high
            zs_low = last_zs.low

            if price > zs_high * (1 + zs_break_gap):
                dir_now = "long"
                kind = "zs_break_up"
                triggered = True
            elif price < zs_low * (1 - zs_break_gap):
                dir_now = "short"
                kind = "zs_break_down"
                triggered = True

        if not triggered or dir_now is None:
            continue

        # ---------- 3）AI 评分 ----------
        score = ai_score_signal(ctx, i, dir_now)
        if score < min_score:
            continue

        # ---------- 4）结构化风险控制 ----------
        sl, tp, rr = compute_risk_for_signal(df_exec, i, dir_now, rr_target=rr_target)
        if rr < 1.2:
            continue

        sig = ChanSignal(
            index=i,
            price=float(price),
            kind=kind,
            direction=dir_now,
            score=score,
            rr=rr,
            sl=sl,
            tp=tp,
        )
        signals.append(sig)

    logging.info(f"[{ctx.symbol}] 生成信号数量（V19_2 结构触发）: {len(signals)}")
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

    # 允许顺序执行信号，但不并行持仓（保守版）
    last_exit_index = -1

    for sig in signals:
        entry_idx = sig.index
        if entry_idx <= last_exit_index or entry_idx >= len(df_exec) - 1:
            continue

        entry_price = sig.price
        direction = sig.direction
        exit_price: Optional[float] = None
        exit_idx: int = entry_idx

        for j in range(entry_idx + 1, min(entry_idx + max_hold_bars, len(df_exec))):
            high = highs[j]
            low = lows[j]

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
        last_exit_index = exit_idx

    total_pnl = sum(t.pnl for t in trades)
    return trades, total_pnl


def run_symbol_v19_2(
    symbol: str,
    days: int,
    intervals: Optional[List[str]] = None,
) -> Dict[str, float]:
    if intervals is None:
        intervals = ["4h", "1h", "15m", "5m"]

    logging.info(f"========== 开始回测 V19_2: {symbol} ==========")
    ctx = build_multilevel_context(symbol, intervals, days)
    df_exec = ctx.interval_dfs["5m"]

    signals = generate_signals_v19_2(ctx, exec_interval="5m")
    trades, total_pnl = backtest_signals(df_exec, signals)

    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl <= 0)

    win_rate = wins / max(1, len(trades))

    logging.info(
        f"[{symbol}] 交易笔数: {len(trades)}, 胜率: {win_rate:.2f}, 总收益: {total_pnl:.4f}"
    )

    return {
        "symbol": symbol,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "pnl": total_pnl,
        "win_rate": win_rate,
    }


# ===================== CLI 入口 =====================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SmartBacktest V19_2 - 多级别缠论结构触发 + AI 评分 回测引擎"
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
            res = run_symbol_v19_2(sym, args.days)
        except FileNotFoundError as e:
            logging.error(f"[{sym}] 回测失败: {e}")
            continue
        except Exception as e:
            logging.exception(f"[{sym}] 回测失败: {e}")
            continue

        total_pnl += res["pnl"]
        total_trades += res["trades"]
        total_wins += res["wins"]

    print("\n========== 📈 V19_2 多级别缠论结构回测战报 ==========")
    print(f"💰 总收益: {total_pnl:.4f}")
    print(f"🔢 总交易数: {total_trades}")
    if total_trades > 0:
        print(f"🎯 综合胜率: {total_wins / total_trades:.2f}")
    else:
        print("🎯 综合胜率: N/A (无交易)")


if __name__ == "__main__":
    main()
