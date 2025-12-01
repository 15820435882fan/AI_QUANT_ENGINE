"""
trend_multi_v1.py

V20_Alpha: 趋势跟踪 + 多因子融合（基础可跑版）

目标：
1）抛开缠论复杂度，先用“经典趋势跟踪 + 多因子评分”做出一套能稳定回测的策略。
2）依托已有本地历史数据（local_data_engine.load_local_kline），快速验证策略效果。
3）为后续实盘上车打基础：信号简单清晰，可扩展。

特性：
- 支持两种模式：trend（趋势跟踪）、multifactor（多因子融合）
- 默认使用 1h K 线（更适合趋势），可通过参数调整
- long / flat 模式（先不做做空，降低复杂度）
"""

import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


@dataclass
class BacktestResult:
    symbol: str
    mode: str
    trades: int
    win_rate: float
    total_return: float
    max_drawdown: float
    sharpe: float


# ===================== 工具函数 =====================


def ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
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


def calc_max_drawdown(equity: pd.Series) -> float:
    """最大回撤（以收益率表示，负数）"""
    cummax = equity.cummax()
    dd = equity / cummax - 1.0
    return float(dd.min())


def calc_sharpe(returns: pd.Series, periods_per_year: int = 365 * 24) -> float:
    """简单 Sharpe：假设每根 K 线为一个 period；1h 则 365*24"""
    if returns.std() == 0 or np.isnan(returns.std()):
        return 0.0
    mean = returns.mean()
    std = returns.std()
    sharpe = (mean * periods_per_year) / (std * np.sqrt(periods_per_year))
    return float(sharpe)


# ===================== 策略 1：趋势跟踪 =====================


def build_trend_strategy(df: pd.DataFrame,
                         fast_window: int = 50,
                         slow_window: int = 200,
                         atr_window: int = 14,
                         atr_mult: float = 2.0) -> pd.DataFrame:
    """
    经典趋势策略（long/flat）：
    - 使用 EMA(fast) 与 EMA(slow) 判断趋势方向
    - price > EMA(fast) 且 EMA(fast) > EMA(slow) → 多头趋势
    - price < EMA(fast) 或 EMA(fast) < EMA(slow) → 空仓
    - ATR 辅助止损宽度（暂不做逐笔止损，只用来衡量波动）
    """
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=fast_window, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow_window, adjust=False).mean()

    # 计算 ATR
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
    df["atr"] = tr.rolling(atr_window).mean()

    # 趋势条件
    df["trend_up"] = (df["ema_fast"] > df["ema_slow"]) & (df["close"] > df["ema_fast"])
    df["position"] = 0
    df.loc[df["trend_up"], "position"] = 1  # long only

    # 为了避免头几行指标缺失，过滤掉 NaN
    df = df.dropna().copy()

    return df


# ===================== 策略 2：多因子融合 =====================


def zscore(series: pd.Series) -> pd.Series:
    mean = series.rolling(100).mean()
    std = series.rolling(100).std()
    z = (series - mean) / (std + 1e-9)
    return z


def build_multifactor_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    多因子 long/flat 策略（简单版）：

    使用的因子：
    1）趋势因子：EMA(50) - EMA(200) / price
    2）动量因子：过去 N 根收益率（例如 24 根）
    3）波动因子：近 N 根收益率波动率（低波动加分）
    4）RSI 因子：RSI > 55 视作偏多

    因子统一 z-score 后加权求和得到 factor_score ∈ [-∞, +∞]
    映射为 position：
        factor_score > 0 → 1
        factor_score ≤ 0 → 0
    """
    df = df.copy()
    close = df["close"]

    # 1) 趋势因子
    ema_fast = close.ewm(span=50, adjust=False).mean()
    ema_slow = close.ewm(span=200, adjust=False).mean()
    trend_raw = (ema_fast - ema_slow) / (close + 1e-9)

    # 2) 动量因子（近 24 根 K 线收益）
    ret = close.pct_change()
    mom_raw = close / close.shift(24) - 1.0

    # 3) 波动因子（取负的波动率：波动越小越好）
    vol_raw = -ret.rolling(48).std()

    # 4) RSI 因子
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    rsi_raw = (rsi - 50) / 50.0  # >0 偏多，<0 偏空

    # 因子 z-score
    f_trend = zscore(trend_raw)
    f_mom = zscore(mom_raw)
    f_vol = zscore(vol_raw)
    f_rsi = zscore(rsi_raw)

    df["f_trend"] = f_trend
    df["f_mom"] = f_mom
    df["f_vol"] = f_vol
    df["f_rsi"] = f_rsi

    # 综合因子评分：可以调整权重
    df["factor_score"] = (
        0.4 * df["f_trend"]
        + 0.3 * df["f_mom"]
        + 0.2 * df["f_vol"]
        + 0.1 * df["f_rsi"]
    )

    # 简单 long/flat 规则
    df["position"] = 0
    df.loc[df["factor_score"] > 0, "position"] = 1

    df = df.dropna().copy()
    return df


# ===================== 回测引擎 =====================


def backtest_long_flat(df: pd.DataFrame) -> BacktestResult:
    """
    long / flat 回测：
    - position ∈ {0,1}
    - 每根 K 线收益：ret * position.shift(1)
    - 统计交易次数、胜率、收益、回撤、Sharpe
    """
    df = df.copy()
    if "position" not in df.columns:
        raise ValueError("DataFrame 中缺少 position 列，请先构建策略信号。")

    # 基础收益
    df["ret"] = df["close"].pct_change()
    df["pos_shift"] = df["position"].shift(1).fillna(0)
    df["strategy_ret"] = df["ret"] * df["pos_shift"]

    equity = (1 + df["strategy_ret"]).cumprod()

    # 交易统计（以 position 的 0→1 变化视为开仓）
    df["pos_change"] = df["position"].diff().fillna(0)
    entries = df[df["pos_change"] > 0].index
    exits = df[df["pos_change"] < 0].index

    # 粗略统计每笔交易盈亏：用开仓到下一次平仓之间的累积 strategy_ret
    trades_pnl: List[float] = []
    if len(entries) > 0:
        # 若最后一次进场后未出现平仓，则以最后一根为平仓
        exits_all = list(exits)
        if len(exits_all) < len(entries):
            exits_all.append(df.index[-1])

        for ent, ex in zip(entries, exits_all):
            sub = df.loc[ent:ex]
            pnl = (1 + sub["strategy_ret"]).prod() - 1.0
            trades_pnl.append(float(pnl))

    trades = len(trades_pnl)
    wins = sum(1 for x in trades_pnl if x > 0)
    win_rate = wins / trades if trades > 0 else 0.0

    total_return = float(equity.iloc[-1] - 1.0)
    max_dd = calc_max_drawdown(equity)
    sharpe = calc_sharpe(df["strategy_ret"].fillna(0))

    return BacktestResult(
        symbol="",
        mode="",
        trades=trades,
        win_rate=win_rate,
        total_return=total_return,
        max_drawdown=max_dd,
        sharpe=sharpe,
    )


# ===================== 运行入口：封装两类策略 =====================


def run_symbol_trend_multi(
    symbol: str,
    days: int,
    interval: str,
    mode: str,
) -> BacktestResult:
    logging.info(f"========== 开始回测 {mode} 策略: {symbol} ({interval}, {days}d) ==========")

    df_raw = load_local_kline(symbol, interval, days)
    df = ensure_ohlc(df_raw)

    if mode == "trend":
        df_sig = build_trend_strategy(df)
    elif mode == "multifactor":
        df_sig = build_multifactor_strategy(df)
    else:
        raise ValueError(f"未知模式: {mode}，支持 'trend' 或 'multifactor'")

    res = backtest_long_flat(df_sig)
    res.symbol = symbol
    res.mode = mode

    logging.info(
        f"[{symbol}][{mode}] 交易数: {res.trades}, "
        f"胜率: {res.win_rate:.2f}, 总收益: {res.total_return:.4f}, "
        f"最大回撤: {res.max_drawdown:.4f}, Sharpe: {res.sharpe:.2f}"
    )
    return res


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="趋势跟踪 + 多因子融合 回测引擎（V20_Alpha）"
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
        default=365,
        help="回测区间天数（对所有周期统一使用）",
    )
    p.add_argument(
        "--interval",
        type=str,
        default="1h",
        help="K线周期，例如: 5m,15m,1h,4h，默认 1h 更适合趋势",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="trend",
        help="策略模式: trend / multifactor",
    )
    return p.parse_args()


def main():
    args = parse_args()
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    mode = args.mode

    total = {
        "trades": 0,
        "total_return": 0.0,
        "max_drawdown": [],
        "sharpe": [],
    }

    results: List[BacktestResult] = []

    for sym in syms:
        try:
            res = run_symbol_trend_multi(sym, args.days, args.interval, mode)
        except FileNotFoundError as e:
            logging.error(f"[{sym}] 回测失败: {e}")
            continue
        except Exception as e:
            logging.exception(f"[{sym}] 回测失败: {e}")
            continue

        results.append(res)
        total["trades"] += res.trades
        total["total_return"] += res.total_return
        total["max_drawdown"].append(res.max_drawdown)
        total["sharpe"].append(res.sharpe)

    print("\n========== 📈 趋势 + 多因子 回测战报 (V20_Alpha) ==========")
    print(f"🧠 模式: {mode}")
    for r in results:
        print(
            f"- {r.symbol}: 交易 {r.trades} 笔 | 胜率 {r.win_rate:.2f} | "
            f"收益 {r.total_return:.4f} | 回撤 {r.max_drawdown:.4f} | Sharpe {r.sharpe:.2f}"
        )

    if results:
        avg_ret = total["total_return"] / len(results)
        avg_dd = np.mean(total["max_drawdown"]) if total["max_drawdown"] else 0.0
        avg_sharpe = np.mean(total["sharpe"]) if total["sharpe"] else 0.0
        print("----------------------------------------------------")
        print(f"📊 平均收益: {avg_ret:.4f}")
        print(f"📉 平均最大回撤: {avg_dd:.4f}")
        print(f"📐 平均 Sharpe: {avg_sharpe:.2f}")
    else:
        print("⚠ 未成功回测任何币种，请检查数据或参数。")


if __name__ == "__main__":
    main()
