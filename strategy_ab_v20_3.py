"""
strategy_ab_v20_3.py

V20_3: 策略A（趋势） + 策略B（多因子）+ 多空 + 杠杆 + ATR 动态止盈止损
       + 交易记录 CSV（Excel 可直接打开） + testdata 目录 + 自动编号

更新要点：
- 输出目录固定为 ./testdata （不存在则自动创建）
- 每次运行为每个 (版本, 策略, 币种) 生成带自增序号的文件：
    V20_3_A_BTCUSDT_001.csv
    V20_3_B_ETHUSDT_001.csv
    ...
- 交易记录包含：
    symbol, strategy, entry_time, exit_time,
    entry_price, exit_price, direction, leverage,
    pnl, pnl_pct, reason
- 初始资金通过代码常量设置：INITIAL_CAPITAL，不再从命令行输入
- 不依赖 openpyxl，仅使用 CSV（Excel 可直接打开）

使用示例（在项目根目录执行）：
    python strategy_ab_v20_3.py --symbols BTCUSDT,ETHUSDT --days 365 --interval 1h
"""

import argparse
import logging
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

# ========== 全局配置 ==========

VERSION_TAG = "V20_3"
OUTPUT_DIR = "testdata"
INITIAL_CAPITAL = 10_000.0  # 初始资金，改这个数字即可

try:
    from local_data_engine import load_local_kline
except Exception:
    def load_local_kline(*args, **kwargs):
        raise RuntimeError("未找到 local_data_engine.load_local_kline，请确认同目录下存在该文件。")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ===================== 数据结构 =====================


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


# ===================== 通用工具函数 =====================


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


# ===================== 占位因子：巨鲸 / 出入金 / 情绪 =====================


def whale_flow_factor_placeholder(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=df.index)


def exchange_flow_factor_placeholder(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=df.index)


def sentiment_factor_placeholder(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=df.index)


# ===================== 策略 A：趋势跟踪 =====================


def build_trend_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    策略A：EMA 趋势强度 → 多空 + 置信度
    - trend_raw = (EMA(50) - EMA(200)) / price
    - signal_dir_A = sign(trend_raw)
    - signal_conf_A = |trend_raw| 标准化后映射到 [0,1]
    """
    df = df.copy()
    close = df["close"]

    ema_fast = close.ewm(span=50, adjust=False).mean()
    ema_slow = close.ewm(span=200, adjust=False).mean()
    trend_raw = (ema_fast - ema_slow) / (close + 1e-9)

    df["trend_raw"] = trend_raw
    df["signal_dir_A"] = np.sign(trend_raw)

    abs_t = trend_raw.abs()
    lo, hi = abs_t.quantile(0.1), abs_t.quantile(0.9)
    span = hi - lo if hi > lo else 1e-9
    conf = (abs_t - lo) / span
    df["signal_conf_A"] = conf.clip(0, 1)

    return df.dropna().copy()


# ===================== 策略 B：多因子 =====================


def build_multifactor_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    策略B：多因子融合（long/short）
    因子：
      - 趋势因子：EMA(50)-EMA(200)
      - 动量因子：24 根收益
      - 波动因子：负波动率（低波动加分）
      - RSI 因子：>50 偏多
      - 巨鲸 / 出入金 / 情绪 因子占位
    """
    df = df.copy()
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

    df["factor_score_B_raw"] = factor_score
    df["signal_dir_B"] = np.sign(factor_score)

    abs_f = factor_score.abs()
    lo, hi = abs_f.quantile(0.1), abs_f.quantile(0.9)
    span = hi - lo if hi > lo else 1e-9
    conf = (abs_f - lo) / span
    df["signal_conf_B"] = conf.clip(0, 1)

    return df.dropna().copy()


# ===================== 文件名自动编号 =====================


def ensure_output_dir() -> None:
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)


def next_sequence_number(symbol: str, strategy_tag: str) -> str:
    """
    扫描 OUTPUT_DIR 下已有的 V20_3_<strategy>_<symbol>_XXX.csv
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


# ===================== 带杠杆 + ATR 风控回测 =====================


def backtest_with_leverage(
    df: pd.DataFrame,
    symbol: str,
    strategy_tag: str,
    sig_dir_col: str,
    sig_conf_col: str,
    base_capital: float = INITIAL_CAPITAL,
    min_leverage: float = 3.0,
    max_leverage: float = 10.0,
    atr_period: int = 14,
    sl_atr_mult: float = 2.5,
    trail_atr_mult: float = 1.5,
    min_conf_threshold: float = 0.25,
    min_hold_bars: int = 3,
) -> Tuple[BacktestResult, pd.DataFrame]:
    """
    多空 + 杠杆 + ATR 止损 + ATR 追踪止盈 + 信号反转
    并记录每笔交易到 trades DataFrame。
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
        res = BacktestResult(symbol, strategy_tag, 0, 0.0, 0.0, 0.0, 0.0, 0.0, "", "")
        return res, trades_df

    equity_vals = [base_capital]
    strategy_rets = [0.0]

    pos_dir = 0            # 当前方向：-1/0/1
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

        # 映射杠杆
        lev = min_leverage + (max_leverage - min_leverage) * sig_conf

        # 先计算当前持仓的本根收益
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

        # 1）风险控制退出：止损 & 追踪止盈
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

        # 2）信号反转或置信度变弱退出（持仓足够长）
        if pos_dir != 0 and bars_since_entry >= min_hold_bars:
            if sig_dir != 0 and sig_dir != pos_dir:
                exit_now = True
                if not exit_reason:
                    exit_reason = "REVERSE"
            elif abs(sig_conf) < min_conf_threshold:
                exit_now = True
                if not exit_reason:
                    exit_reason = "WEAK_SIGNAL"

        # 执行平仓
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

            # 平仓后无仓位
            pos_dir = 0
            pos_lev = 0.0
            entry_price = 0.0
            bars_since_entry = 0

        # 3）开仓条件：空仓 + 信号方向 != 0 + 置信度>=阈值 + ATR 正常
        if pos_dir == 0 and sig_dir != 0 and sig_conf >= min_conf_threshold and cur_atr is not None:
            pos_dir = sig_dir
            pos_lev = lev
            entry_price = price_now
            entry_index = i
            high_since_entry = float(high.iloc[i])
            low_since_entry = float(low.iloc[i])
            bars_since_entry = 0

        equity_vals.append(eq_now)

    # 最后一笔如果仍然有持仓，则在最后一根强制平仓
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

    # 保存 CSV 到 testdata 目录
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


# ===================== A / B 对比运行 =====================


def run_symbol_A_B(
    symbol: str,
    days: int,
    interval: str,
) -> Tuple[BacktestResult, BacktestResult]:
    logging.info(f"========== {symbol}: 加载数据 ({interval}, {days}d) ==========")
    df_raw = load_local_kline(symbol, interval, days)
    df = ensure_ohlc(df_raw)

    # 构建 A 信号
    df_A = build_trend_signals(df)
    df_A["high"] = df["high"]
    df_A["low"] = df["low"]

    # 构建 B 信号
    df_B = build_multifactor_signals(df)
    df_B["high"] = df["high"]
    df_B["low"] = df["low"]

    # 回测 A
    res_A, _ = backtest_with_leverage(
        df_A,
        symbol=symbol,
        strategy_tag="A",
        sig_dir_col="signal_dir_A",
        sig_conf_col="signal_conf_A",
    )

    # 回测 B
    res_B, _ = backtest_with_leverage(
        df_B,
        symbol=symbol,
        strategy_tag="B",
        sig_dir_col="signal_dir_B",
        sig_conf_col="signal_conf_B",
    )

    return res_A, res_B


# ===================== CLI =====================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"策略A(趋势) + 策略B(多因子) 多空杠杆回测引擎 ({VERSION_TAG}, CSV 交易记录版)"
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
    return p.parse_args()


def main():
    args = parse_args()
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    days = args.days
    interval = args.interval

    results_A: List[BacktestResult] = []
    results_B: List[BacktestResult] = []

    for sym in syms:
        try:
            res_A, res_B = run_symbol_A_B(sym, days, interval)
            results_A.append(res_A)
            results_B.append(res_B)
        except FileNotFoundError as e:
            logging.error(f"[{sym}] 回测失败: {e}")
        except Exception as e:
            logging.exception(f"[{sym}] 回测失败: {e}")

    print(f"\n========== 📈 {VERSION_TAG} 回测战报（CSV 交易记录版） ==========")

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

    print_summary("A（趋势）", results_A)
    print_summary("B（多因子）", results_B)


if __name__ == "__main__":
    main()
