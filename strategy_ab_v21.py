"""
strategy_ab_v21.py

V21: 多周期趋势交易系统
--------------------------------
- 策略 A：多周期趋势跟踪（4h 主趋势 + 1h 趋势确认 + 15m 突破入场）
- 策略 B：多周期趋势 + 多因子过滤（在 A 的基础上叠加多因子打分）

共同特性：
- 使用本地 K 线：local_data_engine.load_local_kline(symbol, interval, days)
- 交易执行周期：1h（决策在 1h 上进行）
- 价格与风控：基于 1h K线 (high/low/close)
- 多空、3~10 倍杠杆（由置信度线性映射）
- ATR 止损 + 趋势型追踪止盈（更宽、更耐心）
- 冷静期 + 最小持仓 bars，控制交易次数
- 每笔交易记录保存为 CSV 到 ./testdata 目录下：
    V21_A_BTCUSDT_001.csv
    V21_B_BTCUSDT_001.csv
  可直接用 Excel 打开

用法示例（在项目根目录执行）：
    python strategy_ab_v21.py --symbols BTCUSDT,ETHUSDT --days 365
"""

import argparse
import logging
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

# ========= 全局配置 =========

VERSION_TAG = "V21"
OUTPUT_DIR = "testdata"
INITIAL_CAPITAL = 10_000.0  # 初始资金（USDT）

try:
    from local_data_engine import load_local_kline
except Exception:
    def load_local_kline(*args, **kwargs):
        raise RuntimeError("未找到 local_data_engine.load_local_kline，请确认同目录下存在该文件。")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ========= 数据结构 =========


@dataclass
class BacktestResult:
    symbol: str
    strategy: str
    trades: int
    win_rate: float
    total_return: float
    max_drawdown: float
    sharpe: float
    profit_factor: float
    avg_trade_return: float
    file_path: str


# ========= 工具函数 =========


def ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """确保 DF 有 open/high/low/close，并按时间索引排序。"""
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="ignore")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.set_index("timestamp")
    df = df.sort_index()

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    return df[["open", "high", "low", "close"]].copy()


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def calc_max_drawdown(equity: pd.Series) -> float:
    cummax = equity.cummax()
    dd = equity / cummax - 1.0
    return float(dd.min())


def calc_sharpe(returns: pd.Series, periods_per_year: int = 365 * 24) -> float:
    r = returns.dropna()
    if len(r) < 10:
        return 0.0
    mu, sigma = r.mean(), r.std()
    if sigma == 0 or np.isnan(sigma):
        return 0.0
    return float((mu * periods_per_year) / (sigma * np.sqrt(periods_per_year)))


def zscore(series: pd.Series, window: int = 200) -> pd.Series:
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    return (series - m) / (s + 1e-9)


def ensure_output_dir() -> None:
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)


def next_sequence_number(symbol: str, strategy_tag: str) -> str:
    """
    扫描 OUTPUT_DIR 下已有的 V21_<strategy>_<symbol>_XXX.csv
    找到最大 XXX + 1，返回新的 3 位序号字符串。
    """
    ensure_output_dir()
    prefix = f"{VERSION_TAG}_{strategy_tag}_{symbol}_"
    max_seq = 0
    for fname in os.listdir(OUTPUT_DIR):
        if not fname.startswith(prefix):
            continue
        if not fname.lower().endswith(".csv"):
            continue
        base = fname[:-4]
        parts = base.split("_")
        if len(parts) < 4:
            continue
        seq_str = parts[-1]
        try:
            seq = int(seq_str)
            if seq > max_seq:
                max_seq = seq
        except ValueError:
            continue
    return f"{max_seq + 1:03d}"


# ========= 占位因子：巨鲸 / 出入金 / 情绪 =========


def whale_flow_factor_placeholder(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=df.index)


def exchange_flow_factor_placeholder(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=df.index)


def sentiment_factor_placeholder(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=df.index)


# ========= 多周期特征构建 =========


def build_4h_trend(df_4h: pd.DataFrame) -> pd.DataFrame:
    df = df_4h.copy()
    close = df["close"]
    ema_fast = close.ewm(span=30, adjust=False).mean()
    ema_slow = close.ewm(span=90, adjust=False).mean()
    trend_raw = (ema_fast - ema_slow) / (close + 1e-9)
    df["trend_raw_4h"] = trend_raw
    df["trend_dir_4h"] = np.sign(trend_raw)

    abs_t = trend_raw.abs()
    lo, hi = abs_t.quantile(0.1), abs_t.quantile(0.9)
    span = hi - lo if hi > lo else 1e-9
    strength = ((abs_t - lo) / span).clip(0, 1)
    df["trend_strength_4h"] = strength
    return df[["trend_raw_4h", "trend_dir_4h", "trend_strength_4h"]]


def build_1h_trend(df_1h: pd.DataFrame) -> pd.DataFrame:
    df = df_1h.copy()
    close = df["close"]
    ema_fast = close.ewm(span=50, adjust=False).mean()
    ema_slow = close.ewm(span=150, adjust=False).mean()
    trend_raw = (ema_fast - ema_slow) / (close + 1e-9)
    df["trend_raw_1h"] = trend_raw
    df["trend_dir_1h"] = np.sign(trend_raw)

    abs_t = trend_raw.abs()
    lo, hi = abs_t.quantile(0.1), abs_t.quantile(0.9)
    span = hi - lo if hi > lo else 1e-9
    strength = ((abs_t - lo) / span).clip(0, 1)
    df["trend_strength_1h"] = strength
    return df[["trend_raw_1h", "trend_dir_1h", "trend_strength_1h"]]


def build_15m_breakout(df_15m: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    15m 突破：
    - close > 过去 N 根 high 的最高值 → breakout_up
    - close < 过去 N 根 low 的最低值 → breakout_down
    聚合到 1h 层（按 floor('H')）
    """
    df = df_15m.copy()
    high = df["high"]
    low = df["low"]
    df["hh"] = high.rolling(window).max().shift(1)
    df["ll"] = low.rolling(window).min().shift(1)
    df["breakout_up"] = df["close"] > df["hh"]
    df["breakout_down"] = df["close"] < df["ll"]

    # 映射到 1h
    df["hour_ts"] = df.index.floor("H")
    grp = df.groupby("hour_ts")
    hour_up = grp["breakout_up"].any()
    hour_down = grp["breakout_down"].any()

    breakout_1h = pd.DataFrame({
        "breakout_up_1h": hour_up,
        "breakout_down_1h": hour_down,
    })
    return breakout_1h


def build_multifactor_1h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    在 1h 层面构建多因子评分（用于策略 B）：
    - 趋势因子：EMA(50)-EMA(200)
    - 动量因子：过去 24 根涨幅
    - 波动因子：负的波动率
    - RSI 因子
    - 巨鲸/出入金/情绪占位
    """
    df = df_1h.copy()
    close = df["close"]

    ema_fast = close.ewm(span=50, adjust=False).mean()
    ema_slow = close.ewm(span=200, adjust=False).mean()
    trend_raw = (ema_fast - ema_slow) / (close + 1e-9)

    mom_raw = close / close.shift(24) - 1.0
    vol_raw = -close.pct_change().rolling(48).std()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - 100 / (1 + rs)
    rsi_raw = (rsi - 50) / 50.0

    whale_raw = whale_flow_factor_placeholder(df)
    exflow_raw = exchange_flow_factor_placeholder(df)
    senti_raw = sentiment_factor_placeholder(df)

    f_trend = zscore(trend_raw, window=200)
    f_mom = zscore(mom_raw, window=200)
    f_vol = zscore(vol_raw, window=200)
    f_rsi = zscore(rsi_raw, window=200)
    f_whale = zscore(whale_raw, window=200)
    f_exflow = zscore(exflow_raw, window=200)
    f_senti = zscore(senti_raw, window=200)

    factor_score = (
        0.35 * f_trend
        + 0.25 * f_mom
        + 0.15 * f_vol
        + 0.15 * f_rsi
        + 0.05 * f_whale
        + 0.03 * f_exflow
        + 0.02 * f_senti
    )

    df["factor_score_1h"] = factor_score

    abs_f = factor_score.abs()
    lo, hi = abs_f.quantile(0.1), abs_f.quantile(0.9)
    span = hi - lo if hi > lo else 1e-9
    strength = ((abs_f - lo) / span).clip(0, 1)
    df["factor_strength_1h"] = strength
    df["factor_dir_1h"] = np.sign(factor_score)

    return df[["factor_score_1h", "factor_strength_1h", "factor_dir_1h"]]


# ========= 策略 A：多周期趋势信号 =========


def build_signals_A(df_1h: pd.DataFrame,
                    trend_4h: pd.DataFrame,
                    trend_1h: pd.DataFrame,
                    breakout_1h: pd.DataFrame) -> pd.DataFrame:
    """
    策略 A：
      - 4h 趋势方向 + 强度
      - 1h 趋势方向 + 强度
      - 15m 突破作为入场触发
    生成：
      - signal_dir_A ∈ {-1,0,1}
      - signal_conf_A ∈ [0,1]
    """
    df = df_1h.copy()

    # 对齐 4h 到 1h（向后填充）
    t4 = trend_4h.copy().sort_index()
    base = df.copy()
    base_reset = base.reset_index().rename(columns={base.index.name or "index": "ts"})
    t4_reset = t4.reset_index().rename(columns={t4.index.name or "index": "ts"})
    merged4 = pd.merge_asof(base_reset, t4_reset, on="ts", direction="backward")
    merged4 = merged4.set_index("ts")

    # 1h 趋势直接按索引对齐
    t1 = trend_1h.copy()
    merged = merged4.join(t1, how="left")
    # 15m 突破对齐
    merged = merged.join(breakout_1h, how="left")

    merged["breakout_up_1h"] = merged["breakout_up_1h"].fillna(False)
    merged["breakout_down_1h"] = merged["breakout_down_1h"].fillna(False)

    # 条件：强趋势 + 同向突破
    long_cond = (
        (merged["trend_dir_4h"] == 1)
        & (merged["trend_strength_4h"] >= 0.6)
        & (merged["trend_dir_1h"] == 1)
        & (merged["trend_strength_1h"] >= 0.5)
        & (merged["breakout_up_1h"])
    )

    short_cond = (
        (merged["trend_dir_4h"] == -1)
        & (merged["trend_strength_4h"] >= 0.6)
        & (merged["trend_dir_1h"] == -1)
        & (merged["trend_strength_1h"] >= 0.5)
        & (merged["breakout_down_1h"])
    )

    merged["signal_dir_A"] = 0
    merged.loc[long_cond, "signal_dir_A"] = 1
    merged.loc[short_cond, "signal_dir_A"] = -1

    # 置信度：4h 强度 70% + 1h 强度 30%
    strength = (
        0.7 * merged["trend_strength_4h"].fillna(0)
        + 0.3 * merged["trend_strength_1h"].fillna(0)
    )
    # 没有信号时置信度为 0
    strength[merged["signal_dir_A"] == 0] = 0.0
    merged["signal_conf_A"] = strength.clip(0, 1)

    out = df.copy()
    out["signal_dir_A"] = merged["signal_dir_A"]
    out["signal_conf_A"] = merged["signal_conf_A"]
    return out.dropna().copy()


# ========= 策略 B：多周期趋势 + 多因子 =========


def build_signals_B(df_1h: pd.DataFrame,
                    trend_4h: pd.DataFrame,
                    trend_1h: pd.DataFrame,
                    breakout_1h: pd.DataFrame,
                    multifactor_1h: pd.DataFrame) -> pd.DataFrame:
    """
    策略 B：
      - 与 A 相同的多周期趋势过滤
      - 叠加 1h 多因子评分
    生成：
      - signal_dir_B
      - signal_conf_B
    """
    df = df_1h.copy()

    # 对齐 4h
    t4 = trend_4h.copy().sort_index()
    base = df.copy()
    base_reset = base.reset_index().rename(columns={base.index.name or "index": "ts"})
    t4_reset = t4.reset_index().rename(columns={t4.index.name or "index": "ts"})
    merged4 = pd.merge_asof(base_reset, t4_reset, on="ts", direction="backward")
    merged4 = merged4.set_index("ts")

    t1 = trend_1h.copy()
    merged = merged4.join(t1, how="left")
    merged = merged.join(breakout_1h, how="left")
    merged = merged.join(multifactor_1h, how="left")

    merged["breakout_up_1h"] = merged["breakout_up_1h"].fillna(False)
    merged["breakout_down_1h"] = merged["breakout_down_1h"].fillna(False)

    # 基础趋势过滤：与 A 相同
    trend_long = (
        (merged["trend_dir_4h"] == 1)
        & (merged["trend_strength_4h"] >= 0.5)
        & (merged["trend_dir_1h"] == 1)
        & (merged["trend_strength_1h"] >= 0.4)
    )
    trend_short = (
        (merged["trend_dir_4h"] == -1)
        & (merged["trend_strength_4h"] >= 0.5)
        & (merged["trend_dir_1h"] == -1)
        & (merged["trend_strength_1h"] >= 0.4)
    )

    # 多因子方向
    factor_dir = merged["factor_dir_1h"]
    factor_strength = merged["factor_strength_1h"].fillna(0)

    long_cond = trend_long & (factor_dir > 0) & (factor_strength >= 0.4) & merged["breakout_up_1h"]
    short_cond = trend_short & (factor_dir < 0) & (factor_strength >= 0.4) & merged["breakout_down_1h"]

    merged["signal_dir_B"] = 0
    merged.loc[long_cond, "signal_dir_B"] = 1
    merged.loc[short_cond, "signal_dir_B"] = -1

    # 置信度：0.5 * 4h强度 + 0.3 * 1h强度 + 0.2 * 因子强度
    strength = (
        0.5 * merged["trend_strength_4h"].fillna(0)
        + 0.3 * merged["trend_strength_1h"].fillna(0)
        + 0.2 * factor_strength
    )
    strength[merged["signal_dir_B"] == 0] = 0.0
    merged["signal_conf_B"] = strength.clip(0, 1)

    out = df.copy()
    out["signal_dir_B"] = merged["signal_dir_B"]
    out["signal_conf_B"] = merged["signal_conf_B"]
    return out.dropna().copy()


# ========= 回测（多空 + 杠杆 + 趋势型 ATR 止盈/止损） =========


def backtest_with_leverage(
    df: pd.DataFrame,
    symbol: str,
    strategy_tag: str,
    sig_dir_col: str,
    sig_conf_col: str,
    base_capital: float = INITIAL_CAPITAL,
    min_leverage: float = 3.0,
    max_leverage: float = 10.0,
    atr_period: int = 20,
    sl_atr_mult: float = 3.5,
    trail_atr_mult: float = 3.0,
    min_conf_threshold: float = 0.35,
    min_hold_bars: int = 12,
) -> Tuple[BacktestResult, pd.DataFrame]:
    """
    以 1h 为执行周期：
      - 多空 + 杠杆
      - ATR 止损 + 趋势型追踪止盈（倍数更大，更耐心）
      - 冷静期：最小持仓 bars
    """
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    atr = calc_atr(df, period=atr_period)
    df["atr"] = atr

    n = len(df)
    if n < 50:
        trades_df = pd.DataFrame(
            columns=[
                "symbol", "strategy", "entry_time", "exit_time",
                "entry_price", "exit_price", "direction", "leverage",
                "pnl", "pnl_pct", "reason",
            ]
        )
        res = BacktestResult(
            symbol=symbol,
            strategy=strategy_tag,
            trades=0,
            win_rate=0.0,
            total_return=0.0,
            max_drawdown=0.0,
            sharpe=0.0,
            profit_factor=0.0,
            avg_trade_return=0.0,
            file_path="",
        )
        return res, trades_df

    equity_vals = [base_capital]
    strategy_rets = [0.0]

    pos_dir = 0
    pos_lev = 0.0
    entry_price = 0.0
    entry_index = 0
    high_since_entry = 0.0
    low_since_entry = 0.0
    bars_since_entry = 0

    trades_records: List[dict] = []

    for i in range(1, n):
        idx_now = df.index[i]
        price_prev = float(close.iloc[i - 1])
        price_now = float(close.iloc[i])

        sig_dir = int(df[sig_dir_col].iloc[i])
        sig_conf = float(df[sig_conf_col].iloc[i])
        cur_atr = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else None

        lev = min_leverage + (max_leverage - min_leverage) * sig_conf

        eq_prev = equity_vals[-1]
        if pos_dir != 0:
            ret = price_now / price_prev - 1.0
            eq_now = eq_prev * (1.0 + ret * pos_dir * pos_lev)
            strategy_rets.append(ret * pos_dir * pos_lev)
            bars_since_entry += 1
            high_since_entry = max(high_since_entry, float(high.iloc[i]))
            low_since_entry = min(low_since_entry, float(low.iloc[i]))
        else:
            eq_now = eq_prev
            strategy_rets.append(0.0)

        exit_now = False
        exit_reason = ""

        # 1）风险控制：止损 & 趋势型追踪止盈
        if pos_dir != 0 and cur_atr is not None:
            if pos_dir > 0:
                sl = entry_price - sl_atr_mult * cur_atr
                trail = high_since_entry - trail_atr_mult * cur_atr
                if float(low.iloc[i]) <= sl:
                    exit_now = True
                    exit_reason = "SL"
                elif float(low.iloc[i]) <= trail and bars_since_entry >= min_hold_bars:
                    exit_now = True
                    exit_reason = "TRAIL"
            else:
                sl = entry_price + sl_atr_mult * cur_atr
                trail = low_since_entry + trail_atr_mult * cur_atr
                if float(high.iloc[i]) >= sl:
                    exit_now = True
                    exit_reason = "SL"
                elif float(high.iloc[i]) >= trail and bars_since_entry >= min_hold_bars:
                    exit_now = True
                    exit_reason = "TRAIL"

        # 2）信号反转 / 置信度减弱（仅在持仓足够久）
        if pos_dir != 0 and bars_since_entry >= min_hold_bars:
            if sig_dir != 0 and sig_dir != pos_dir:
                exit_now = True
                if not exit_reason:
                    exit_reason = "REVERSE"
            elif sig_conf < min_conf_threshold:
                exit_now = True
                if not exit_reason:
                    exit_reason = "WEAK_SIGNAL"

        # 平仓
        if pos_dir != 0 and exit_now:
            if pos_dir > 0:
                pnl_pct = (price_now / entry_price - 1.0) * pos_lev
            else:
                pnl_pct = (entry_price / price_now - 1.0) * pos_lev
            pnl = base_capital * pnl_pct

            trades_records.append(
                {
                    "symbol": symbol,
                    "strategy": strategy_tag,
                    "entry_time": df.index[entry_index],
                    "exit_time": idx_now,
                    "entry_price": entry_price,
                    "exit_price": price_now,
                    "direction": pos_dir,
                    "leverage": pos_lev,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "reason": exit_reason,
                }
            )

            pos_dir = 0
            pos_lev = 0.0
            entry_price = 0.0
            bars_since_entry = 0

        # 开仓：空仓 + 强信号 + ATR 正常
        if pos_dir == 0 and sig_dir != 0 and sig_conf >= min_conf_threshold and cur_atr is not None:
            pos_dir = sig_dir
            pos_lev = lev
            entry_price = price_now
            entry_index = i
            high_since_entry = float(high.iloc[i])
            low_since_entry = float(low.iloc[i])
            bars_since_entry = 0

        equity_vals.append(eq_now)

    # 最后一笔强制平仓
    if pos_dir != 0:
        final_price = float(close.iloc[-1])
        if pos_dir > 0:
            pnl_pct = (final_price / entry_price - 1.0) * pos_lev
        else:
            pnl_pct = (entry_price / final_price - 1.0) * pos_lev
        pnl = base_capital * pnl_pct

        trades_records.append(
            {
                "symbol": symbol,
                "strategy": strategy_tag,
                "entry_time": df.index[entry_index],
                "exit_time": df.index[-1],
                "entry_price": entry_price,
                "exit_price": final_price,
                "direction": pos_dir,
                "leverage": pos_lev,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "reason": "END",
            }
        )

    equity = pd.Series(equity_vals, index=df.index)
    ret_series = pd.Series(strategy_rets, index=df.index)
    trades_df = pd.DataFrame(trades_records)

    total_return = float(equity.iloc[-1] / base_capital - 1.0)
    max_dd = calc_max_drawdown(equity)
    sharpe = calc_sharpe(ret_series)

    trades_count = len(trades_df)
    if trades_count > 0:
        wins = trades_df[trades_df["pnl"] > 0]
        wins_sum = wins["pnl"].sum()
        losses = trades_df[trades_df["pnl"] < 0]
        losses_sum = losses["pnl"].sum()
        win_rate = len(wins) / trades_count
        if losses_sum < 0:
            profit_factor = wins_sum / abs(losses_sum)
        else:
            profit_factor = float("inf") if wins_sum > 0 else 0.0
        avg_trade_return = trades_df["pnl_pct"].mean()
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_return = 0.0

    # 保存 CSV
    ensure_output_dir()
    seq = next_sequence_number(symbol, strategy_tag)
    fname = f"{VERSION_TAG}_{strategy_tag}_{symbol}_{seq}.csv"
    fpath = os.path.join(OUTPUT_DIR, fname)
    trades_df.to_csv(fpath, index=False, encoding="utf-8-sig")

    res = BacktestResult(
        symbol=symbol,
        strategy=strategy_tag,
        trades=trades_count,
        win_rate=float(win_rate),
        total_return=total_return,
        max_drawdown=max_dd,
        sharpe=sharpe,
        profit_factor=float(profit_factor),
        avg_trade_return=float(avg_trade_return),
        file_path=fpath,
    )
    return res, trades_df


# ========= 单币种运行：加载多周期，构建 A/B，回测 =========


def run_symbol_A_B(symbol: str, days: int) -> Tuple[BacktestResult, BacktestResult]:
    logging.info(f"========== {symbol}: 加载多周期数据 (4h/1h/15m, {days}d) ==========")

    # 1h 作为主执行周期
    df_1h_raw = load_local_kline(symbol, "1h", days)
    df_1h = ensure_ohlc(df_1h_raw)

    # 4h & 15m
    df_4h_raw = load_local_kline(symbol, "4h", days + 30)  # 多取一点，便于趋势稳定
    df_4h = ensure_ohlc(df_4h_raw)

    df_15m_raw = load_local_kline(symbol, "15m", days)
    df_15m = ensure_ohlc(df_15m_raw)

    # 确保 15m 有 high/low
    if "high" not in df_15m.columns or "low" not in df_15m.columns:
        df_15m["high"] = df_15m["close"]
        df_15m["low"] = df_15m["close"]

    # 构建多周期特征
    trend4 = build_4h_trend(df_4h)
    trend1 = build_1h_trend(df_1h)
    breakout1 = build_15m_breakout(df_15m)
    factor1 = build_multifactor_1h(df_1h)

    # 构建策略 A / B 信号
    df_A = build_signals_A(df_1h, trend4, trend1, breakout1)
    # 确保 high/low 在 df_A
    df_A["high"] = df_1h["high"]
    df_A["low"] = df_1h["low"]

    df_B = build_signals_B(df_1h, trend4, trend1, breakout1, factor1)
    df_B["high"] = df_1h["high"]
    df_B["low"] = df_1h["low"]

    # 回测
    res_A, _ = backtest_with_leverage(
        df_A,
        symbol=symbol,
        strategy_tag="A",
        sig_dir_col="signal_dir_A",
        sig_conf_col="signal_conf_A",
    )

    res_B, _ = backtest_with_leverage(
        df_B,
        symbol=symbol,
        strategy_tag="B",
        sig_dir_col="signal_dir_B",
        sig_conf_col="signal_conf_B",
    )

    logging.info(
        f"[{symbol}][A] Trades={res_A.trades}, WinRate={res_A.win_rate:.2f}, "
        f"Ret={res_A.total_return:.4f}, DD={res_A.max_drawdown:.4f}, Sharpe={res_A.sharpe:.2f}"
    )
    logging.info(
        f"[{symbol}][B] Trades={res_B.trades}, WinRate={res_B.win_rate:.2f}, "
        f"Ret={res_B.total_return:.4f}, DD={res_B.max_drawdown:.4f}, Sharpe={res_B.sharpe:.2f}"
    )

    return res_A, res_B


# ========= CLI & 主入口 =========


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"多周期趋势系统 ({VERSION_TAG}) - 策略A(趋势) & 策略B(多因子)"
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
    return p.parse_args()


def main():
    args = parse_args()
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    days = args.days

    results_A: List[BacktestResult] = []
    results_B: List[BacktestResult] = []

    for sym in syms:
        try:
            res_A, res_B = run_symbol_A_B(sym, days)
            results_A.append(res_A)
            results_B.append(res_B)
        except FileNotFoundError as e:
            logging.error(f"[{sym}] 回测失败: {e}")
        except Exception as e:
            logging.exception(f"[{sym}] 回测失败: {e}")

    print(f"\n========== 📈 {VERSION_TAG} 多周期趋势系统 回测战报 ==========")

    def print_summary(tag: str, res_list: List[BacktestResult]):
        print(f"\n🧠 策略 {tag}:")
        if not res_list:
            print("  ⚠ 无成功回测记录。")
            return
        total_ret = 0.0
        total_dd = []
        total_sharpe = []
        total_pf = []
        total_trades = 0
        total_wins = 0
        for r in res_list:
            print(
                f"- {r.symbol}: "
                f"Trades={r.trades}, WinRate={r.win_rate:.2f}, "
                f"PF={r.profit_factor:.2f}, Sharpe={r.sharpe:.2f}, "
                f"MaxDD={r.max_drawdown:.4f}, TotalRet={r.total_return:.4f}, "
                f"AvgTradeRet={r.avg_trade_return:.4f}, "
                f"File={r.file_path}"
            )
            total_ret += r.total_return
            total_dd.append(r.max_drawdown)
            total_sharpe.append(r.sharpe)
            total_pf.append(r.profit_factor)
            total_trades += r.trades
            total_wins += int(r.trades * r.win_rate)

        n = len(res_list)
        avg_ret = total_ret / n
        avg_dd = float(np.mean(total_dd))
        avg_sh = float(np.mean(total_sharpe))
        avg_pf = float(np.mean(total_pf))
        win_rate_all = total_wins / total_trades if total_trades > 0 else 0.0
        print("  ----------------------------------------------------")
        print(
            f"  📊 平均收益: {avg_ret:.4f} | 平均回撤: {avg_dd:.4f} | "
            f"平均 Sharpe: {avg_sh:.2f} | 平均 PF: {avg_pf:.2f} | "
            f"综合胜率: {win_rate_all:.2f}"
        )

    print_summary("A（多周期趋势）", results_A)
    print_summary("B（多周期趋势 + 多因子）", results_B)


if __name__ == "__main__":
    main()
