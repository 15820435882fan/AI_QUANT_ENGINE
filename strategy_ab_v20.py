"""
strategy_ab_v20.py

V20: 趋势跟踪（策略A）+ 多因子融合（策略B） + 杠杆 + 多空 + 动态止盈止损（基础可跑版）

说明：
- 依赖 local_data_engine.load_local_kline(symbol, interval, days) 读取本地 K 线
- 使用 1h K 线为默认周期（更适合趋势），支持多空、3~10 倍杠杆
- 杠杆大小基于“信号置信度”（0~1）线性映射
- 动态止盈止损：ATR 止损 + ATR 追踪止损，让盈利在趋势中奔跑、极端行情自动止损
- 策略A：趋势跟踪（EMA 快慢+趋势强度）
- 策略B：多因子融合（趋势+动量+波动+RSI+预留：巨鲸/资金流/情绪因子）

注意：
- 目前回测环境中没有“巨鲸/出入金/情绪”真实历史数据，这里预留因子接口，
  回测阶段先用 0 或简单 proxy，未来可接链上&行情API补足。
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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ===================== 数据结构 =====================


@dataclass
class Trade:
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    direction: int        # +1 / -1
    leverage: float


@dataclass
class BacktestResult:
    symbol: str
    strategy: str
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


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr


def calc_max_drawdown(equity: pd.Series) -> float:
    cummax = equity.cummax()
    dd = equity / cummax - 1.0
    return float(dd.min())


def calc_sharpe(returns: pd.Series, periods_per_year: int = 365 * 24) -> float:
    r = returns.dropna()
    if len(r) < 10:
        return 0.0
    mu = r.mean()
    sigma = r.std()
    if sigma == 0 or np.isnan(sigma):
        return 0.0
    # 年化（假设每根K线一个period）
    return float((mu * periods_per_year) / (sigma * np.sqrt(periods_per_year)))


def zscore(series: pd.Series, window: int = 200) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    z = (series - mean) / (std + 1e-9)
    return z


# ===================== 预留因子：巨鲸 / 资金流 / 情绪 =====================


def whale_flow_factor_placeholder(df: pd.DataFrame) -> pd.Series:
    """
    巨鲸监控因子占位：
    理想情况：
        - 读链上大额转账统计（>X万美元）
        - 计算过去N小时内大额净买入/净卖出方向和强度
    当前回测暂时没有这些数据，这里返回 0 序列，将来可以接真实数据。
    """
    return pd.Series(0.0, index=df.index)


def exchange_flow_factor_placeholder(df: pd.DataFrame) -> pd.Series:
    """
    交易所出入金因子占位：
    理想情况：
        - 读交易所储备变动、净流入/净流出
    当前先返回 0。
    """
    return pd.Series(0.0, index=df.index)


def sentiment_factor_placeholder(df: pd.DataFrame) -> pd.Series:
    """
    媒体情绪因子占位：
    理想情况：
        - 抓取新闻/推特/telegram 情绪指标
    当前先返回 0。
    """
    return pd.Series(0.0, index=df.index)


# ===================== 策略A：趋势跟踪 =====================


def build_trend_signals(
    df: pd.DataFrame,
    fast_window: int = 50,
    slow_window: int = 200,
) -> pd.DataFrame:
    """
    策略A：趋势跟踪

    - 使用 EMA(fast) / EMA(slow) 构造趋势强度
    - 趋势方向：sign(trend_strength)
    - 趋势置信度：|trend_strength| 映射到 [0,1]
    - 支持多空
    """
    df = df.copy()
    close = df["close"]

    ema_fast = close.ewm(span=fast_window, adjust=False).mean()
    ema_slow = close.ewm(span=slow_window, adjust=False).mean()

    trend_raw = (ema_fast - ema_slow) / (close + 1e-9)  # 相对差值
    df["trend_raw"] = trend_raw

    # 方向：正多负空
    df["signal_dir_A"] = 0
    df.loc[trend_raw > 0, "signal_dir_A"] = 1
    df.loc[trend_raw < 0, "signal_dir_A"] = -1

    # 置信度：|trend_raw| 按百分位裁剪
    abs_trend = trend_raw.abs()
    hi = abs_trend.quantile(0.9)
    lo = abs_trend.quantile(0.1)
    span = max(1e-9, hi - lo)
    conf = (abs_trend - lo) / span
    conf = conf.clip(lower=0, upper=1)
    df["signal_conf_A"] = conf

    df = df.dropna().copy()
    return df


# ===================== 策略B：多因子融合 =====================


def build_multifactor_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    策略B：多因子融合（long/short）

    因子：
    1）趋势因子：EMA(50)-EMA(200)
    2）动量因子：近24根价格变化
    3）波动因子：负的波动率（低波动更好）
    4）RSI因子：>50偏多，<50偏空
    5）巨鲸/资金流/情绪因子：占位，未来接入数据

    综合：
        factor_score_B = 加权求和(z-score因子)
        signal_dir_B = sign(factor_score_B)
        signal_conf_B = |factor_score_B| 映射到 [0,1]
    """
    df = df.copy()
    close = df["close"]

    # 1) 趋势因子
    ema_fast = close.ewm(span=50, adjust=False).mean()
    ema_slow = close.ewm(span=200, adjust=False).mean()
    trend_raw = (ema_fast - ema_slow) / (close + 1e-9)

    # 2) 动量因子
    mom_raw = close / close.shift(24) - 1.0

    # 3) 波动因子（低波动加分）
    ret = close.pct_change()
    vol_raw = -ret.rolling(48).std()

    # 4) RSI 因子
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    rsi_raw = (rsi - 50) / 50.0

    # 5) 占位因子
    whale_raw = whale_flow_factor_placeholder(df)
    exflow_raw = exchange_flow_factor_placeholder(df)
    senti_raw = sentiment_factor_placeholder(df)

    # z-score 各因子
    f_trend = zscore(trend_raw, window=200)
    f_mom = zscore(mom_raw, window=200)
    f_vol = zscore(vol_raw, window=200)
    f_rsi = zscore(rsi_raw, window=200)
    f_whale = zscore(whale_raw, window=200)
    f_exflow = zscore(exflow_raw, window=200)
    f_senti = zscore(senti_raw, window=200)

    # 组合因子（权重可以后续调参）
    factor_score = (
        0.35 * f_trend
        + 0.25 * f_mom
        + 0.15 * f_vol
        + 0.15 * f_rsi
        + 0.05 * f_whale
        + 0.03 * f_exflow
        + 0.02 * f_senti
    )

    df["factor_score_B_raw"] = factor_score

    # 方向：正多负空
    df["signal_dir_B"] = 0
    df.loc[factor_score > 0, "signal_dir_B"] = 1
    df.loc[factor_score < 0, "signal_dir_B"] = -1

    # 置信度：|factor_score| 映射到 [0,1]
    abs_score = factor_score.abs()
    hi = abs_score.quantile(0.9)
    lo = abs_score.quantile(0.1)
    span = max(1e-9, hi - lo)
    conf = (abs_score - lo) / span
    conf = conf.clip(lower=0, upper=1)
    df["signal_conf_B"] = conf

    df = df.dropna().copy()
    return df


# ===================== 带杠杆 & 动态止损的回测 =====================


def backtest_with_leverage(
    df: pd.DataFrame,
    signal_dir_col: str,
    signal_conf_col: str,
    base_capital: float = 10_000.0,
    min_leverage: float = 3.0,
    max_leverage: float = 10.0,
    atr_period: int = 14,
    sl_atr_mult: float = 2.5,
    trail_atr_mult: float = 1.5,
) -> Tuple[BacktestResult, pd.Series]:
    """
    多空 + 杠杆 + ATR 止损 + ATR 追踪止盈

    - signal_dir ∈ {-1,0,1}
    - signal_conf ∈ [0,1] 映射到 [min_leverage, max_leverage]
    - 单笔仓位基于 close[t] → 持仓方向 & 杠杆
    - 每根K线更新：
        equity *= (1 + ret * dir * leverage)
      其中 ret = close_t / close_{t-1} - 1
    - 止损：
        入场时记录 entry_price，计算 ATR(t)，
        long: SL = entry - sl_atr_mult * ATR
        short: SL = entry + sl_atr_mult * ATR
    - 追踪止盈：
        long: 记录最高价 high_since_entry，trail = high_since_entry - trail_atr_mult * ATR
        short: 记录最低价 low_since_entry， trail = low_since_entry + trail_atr_mult * ATR
    - 信号反向时也可触发平仓并反手（简化：先平后开）
    """
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    atr = calc_atr(df, period=atr_period)
    df["atr"] = atr

    # 准备遍历
    n = len(df)
    if n < 50:
        # 数据太少
        equity = pd.Series([base_capital] * n, index=df.index)
        res = BacktestResult(
            symbol="",
            strategy="",
            trades=0,
            win_rate=0.0,
            total_return=0.0,
            max_drawdown=0.0,
            sharpe=0.0,
        )
        return res, equity

    equity_vals = [base_capital]
    trades: List[Trade] = []

    # 当前持仓状态
    pos_dir = 0         # -1 / 0 / +1
    pos_lev = 0.0
    entry_price = 0.0
    entry_index = 0
    high_since_entry = 0.0
    low_since_entry = 0.0

    # 用于计算每根策略收益（Sharpe）
    strategy_rets: List[float] = [0.0]

    for i in range(1, n):
        price_prev = close.iloc[i - 1]
        price_now = close.iloc[i]
        idx_now = df.index[i]

        sig_dir = int(df[signal_dir_col].iloc[i])
        sig_conf = float(df[signal_conf_col].iloc[i])

        # 映射杠杆
        leverage = min_leverage + (max_leverage - min_leverage) * sig_conf

        # 先根据当前持仓，计算本根收益
        eq_prev = equity_vals[-1]
        if pos_dir != 0:
            ret = (price_now / price_prev) - 1.0
            eq_now = eq_prev * (1.0 + ret * pos_dir * pos_lev)
            strategy_ret = ret * pos_dir * pos_lev
        else:
            eq_now = eq_prev
            strategy_ret = 0.0

        # 更新追踪高低价
        if pos_dir != 0:
            high_since_entry = max(high_since_entry, high.iloc[i])
            low_since_entry = min(low_since_entry, low.iloc[i])

        exit_now = False
        exit_reason = ""

        # 止损/追踪止盈触发
        current_atr = atr.iloc[i]
        if pos_dir != 0 and not np.isnan(current_atr):
            if pos_dir > 0:
                sl = entry_price - sl_atr_mult * current_atr
                trail = high_since_entry - trail_atr_mult * current_atr
                if low.iloc[i] <= sl:
                    exit_now = True
                    exit_reason = "SL"
                elif low.iloc[i] <= trail:
                    exit_now = True
                    exit_reason = "TRAIL"
            else:
                sl = entry_price + sl_atr_mult * current_atr
                trail = low_since_entry + trail_atr_mult * current_atr
                if high.iloc[i] >= sl:
                    exit_now = True
                    exit_reason = "SL"
                elif high.iloc[i] >= trail:
                    exit_now = True
                    exit_reason = "TRAIL"

        # 信号反向也视作出场条件
        if pos_dir != 0 and sig_dir != 0 and sig_dir != pos_dir:
            exit_now = True
            exit_reason = exit_reason or "REVERSE"

        # 执行平仓
        if pos_dir != 0 and exit_now:
            trades.append(
                Trade(
                    entry_index=entry_index,
                    exit_index=i,
                    entry_price=entry_price,
                    exit_price=float(price_now),
                    direction=pos_dir,
                    leverage=pos_lev,
                )
            )
            pos_dir = 0
            pos_lev = 0.0
            entry_price = 0.0

        # 平仓后 equity 已经是 eq_now（包含本根收益），继续考虑是否开仓
        # or 若原本空仓，再考虑开仓
        if pos_dir == 0 and sig_dir != 0 and not np.isnan(current_atr):
            # 开新仓
            pos_dir = sig_dir
            pos_lev = leverage
            entry_price = float(price_now)
            entry_index = i
            high_since_entry = high.iloc[i]
            low_since_entry = low.iloc[i]

        equity_vals.append(eq_now)
        strategy_rets.append(strategy_ret)

    equity = pd.Series(equity_vals, index=df.index)
    returns = pd.Series(strategy_rets, index=df.index)

    # 统计
    total_return = float(equity.iloc[-1] / base_capital - 1.0)
    max_dd = calc_max_drawdown(equity)
    sharpe = calc_sharpe(returns)

    wins = 0
    for t in trades:
        if t.direction > 0:
            pnl = (t.exit_price / t.entry_price - 1.0) * t.leverage
        else:
            pnl = (t.entry_price / t.exit_price - 1.0) * t.leverage
        if pnl > 0:
            wins += 1
    trades_count = len(trades)
    win_rate = wins / trades_count if trades_count > 0 else 0.0

    result = BacktestResult(
        symbol="",
        strategy="",
        trades=trades_count,
        win_rate=win_rate,
        total_return=total_return,
        max_drawdown=max_dd,
        sharpe=sharpe,
    )
    return result, equity


# ===================== 统一运行：策略A & 策略B 对比 =====================


def run_symbol_A_B(
    symbol: str,
    days: int,
    interval: str = "1h",
    base_capital: float = 10_000.0,
) -> Tuple[BacktestResult, BacktestResult]:
    logging.info(f"========== {symbol}: 加载数据 ({interval}, {days}d) ==========")
    df_raw = load_local_kline(symbol, interval, days)
    df = ensure_ohlc(df_raw)

    # 构建策略A信号
    df_A = build_trend_signals(df)
    # 与 df 对齐 ATR 等
    df_A = df_A.join(df[["high", "low"]], how="left")

    # 构建策略B信号
    df_B = build_multifactor_signals(df)
    df_B = df_B.join(df[["high", "low"]], how="left")

    # 回测策略A
    res_A, equity_A = backtest_with_leverage(
        df_A,
        signal_dir_col="signal_dir_A",
        signal_conf_col="signal_conf_A",
        base_capital=base_capital,
    )
    res_A.symbol = symbol
    res_A.strategy = "A_trend"

    # 回测策略B
    res_B, equity_B = backtest_with_leverage(
        df_B,
        signal_dir_col="signal_dir_B",
        signal_conf_col="signal_conf_B",
        base_capital=base_capital,
    )
    res_B.symbol = symbol
    res_B.strategy = "B_multifactor"

    logging.info(
        f"[{symbol}][A] 交易数: {res_A.trades}, 胜率: {res_A.win_rate:.2f}, "
        f"收益: {res_A.total_return:.4f}, 回撤: {res_A.max_drawdown:.4f}, Sharpe: {res_A.sharpe:.2f}"
    )
    logging.info(
        f"[{symbol}][B] 交易数: {res_B.trades}, 胜率: {res_B.win_rate:.2f}, "
        f"收益: {res_B.total_return:.4f}, 回撤: {res_B.max_drawdown:.4f}, Sharpe: {res_B.sharpe:.2f}"
    )

    return res_A, res_B


# ===================== CLI =====================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="策略A(趋势) + 策略B(多因子) 多空杠杆回测引擎 (V20)"
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
        help="回测区间天数",
    )
    p.add_argument(
        "--interval",
        type=str,
        default="1h",
        help="K线周期，例如: 1h,4h 等，默认 1h",
    )
    p.add_argument(
        "--capital",
        type=float,
        default=10_000.0,
        help="初始资金（USDT），默认 10000",
    )
    return p.parse_args()


def main():
    args = parse_args()
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    days = args.days
    interval = args.interval
    capital = args.capital

    results_A: List[BacktestResult] = []
    results_B: List[BacktestResult] = []

    for sym in syms:
        try:
            res_A, res_B = run_symbol_A_B(sym, days, interval, base_capital=capital)
        except FileNotFoundError as e:
            logging.error(f"[{sym}] 回测失败: {e}")
            continue
        except Exception as e:
            logging.exception(f"[{sym}] 回测失败: {e}")
            continue

        results_A.append(res_A)
        results_B.append(res_B)

    print("\n========== 📈 V20 策略A(趋势) & 策略B(多因子) 回测战报 ==========")
    for res_list, tag in [(results_A, "A_trend"), (results_B, "B_multifactor")]:
        print(f"\n🧠 策略 {tag}:")
        total_ret = 0.0
        total_dd: List[float] = []
        total_sharpe: List[float] = []
        total_trades = 0
        total_wins = 0

        for r in res_list:
            print(
                f"- {r.symbol}: 交易 {r.trades} 笔 | 胜率 {r.win_rate:.2f} | "
                f"收益 {r.total_return:.4f} | 回撤 {r.max_drawdown:.4f} | Sharpe {r.sharpe:.2f}"
            )
            total_ret += r.total_return
            total_dd.append(r.max_drawdown)
            total_sharpe.append(r.sharpe)
            total_trades += r.trades
            total_wins += int(r.trades * r.win_rate)

        if res_list:
            n = len(res_list)
            avg_ret = total_ret / n
            avg_dd = float(np.mean(total_dd)) if total_dd else 0.0
            avg_sh = float(np.mean(total_sharpe)) if total_sharpe else 0.0
            overall_win = total_wins / total_trades if total_trades > 0 else 0.0

            print("----------------------------------------------------")
            print(f"📊 平均收益: {avg_ret:.4f}")
            print(f"📉 平均最大回撤: {avg_dd:.4f}")
            print(f"📐 平均 Sharpe: {avg_sh:.2f}")
            print(f"🎯 综合胜率: {overall_win:.2f}")
        else:
            print("⚠ 没有成功回测的品种。")


if __name__ == "__main__":
    main()
