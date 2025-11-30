import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ===================== 日志配置 =====================

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("SmartBacktest")
    if logger.handlers:  # 避免重复添加 handler
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - SmartBacktest - INFO - %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


# ===================== 数据生成（纯模拟） =====================

def generate_synthetic_data(symbol: str, days: int, seed: int) -> pd.DataFrame:
    """
    简单生成 5m 级别的模拟 K 线数据，用于策略回测。
    """
    rng = np.random.default_rng(seed)
    bars_per_day = 24 * 12  # 5m 一天 288 根
    n = days * bars_per_day

    # 不同币种给一个不同的基准价格和波动
    base_price_map = {
        "BTC/USDT": (30000.0, 0.015),
        "ETH/USDT": (2000.0, 0.018),
        "SOL/USDT": (50.0, 0.03),
    }
    base_price, sigma = base_price_map.get(symbol, (100.0, 0.02))

    prices = np.zeros(n)
    prices[0] = base_price
    for i in range(1, n):
        # 简单随机游走
        ret = rng.normal(0.0, sigma)
        prices[i] = max(0.1, prices[i - 1] * (1.0 + ret))

    # 生成 OHLCV
    df = pd.DataFrame(index=pd.RangeIndex(n), data={"close": prices})
    noise = rng.normal(0.0, sigma * 0.3, size=n)

    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1) * (1 + np.abs(noise))
    df["low"] = df[["open", "close"]].min(axis=1) * (1 - np.abs(noise))
    df["volume"] = rng.uniform(10.0, 1000.0, size=n)

    return df[["open", "high", "low", "close", "volume"]]


# ===================== 技术指标 =====================

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # EMA 趋势
    ema_fast = close.ewm(span=20, adjust=False).mean()
    ema_slow = close.ewm(span=60, adjust=False).mean()
    ema_diff = (ema_fast - ema_slow) / ema_slow

    # ATR 波动率
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()
    atr_pct = atr / close

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    out = df.copy()
    out["ema_diff"] = ema_diff
    out["atr_pct"] = atr_pct
    out["rsi"] = rsi
    return out


# ===================== 统计结构体 =====================

@dataclass
class SymbolStats:
    trades: int = 0
    wins: int = 0
    pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    blocked_trend: int = 0
    blocked_vol: int = 0
    blocked_rsi: int = 0


# ===================== 单币种回测核心 =====================

def run_backtest_for_symbol(
    df: pd.DataFrame,
    symbol: str,
    initial_capital: float,
    logger: logging.Logger,
) -> Tuple[SymbolStats, float]:
    """
    对单个币种执行回测。
    返回 (统计结果, 最终资金)
    """
    stats = SymbolStats()
    if df.empty:
        return stats, initial_capital

    df = compute_indicators(df)

    # 动态阈值根据波动率适配
    atr_pct = df["atr_pct"].dropna()
    if atr_pct.empty:
        return stats, initial_capital

    median_atr = float(atr_pct.median())
    # 趋势阈值，随波动率浮动
    trend_thr = min(max(median_atr * 0.6, 0.0008), 0.003)
    # 波动率合理区间
    min_atr_pct = max(0.0005, median_atr * 0.2)
    max_atr_pct = min(0.02, median_atr * 3.0)

    equity = initial_capital
    peak_equity = equity
    position = 0  # 1=多, -1=空, 0=空仓
    entry_price = 0.0
    position_size = 0.0

    # 简单固定每次用 10% 资金做一笔
    risk_fraction = 0.1

    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row["close"])
        ed = float(row["ema_diff"]) if pd.notna(row["ema_diff"]) else np.nan
        volp = float(row["atr_pct"]) if pd.notna(row["atr_pct"]) else np.nan
        rsi = float(row["rsi"]) if pd.notna(row["rsi"]) else np.nan

        if np.isnan(ed) or np.isnan(volp) or np.isnan(rsi):
            continue

        # 1) 波动率过滤：太低 or 太高都不要
        if not (min_atr_pct <= volp <= max_atr_pct):
            stats.blocked_vol += 1
            continue

        # 2) 趋势过滤：太弱 or 太强都过滤掉一部分
        strong_trend = abs(ed) > trend_thr * 2.5
        weak_trend = abs(ed) < trend_thr * 0.4
        if strong_trend or weak_trend:
            stats.blocked_trend += 1
            continue

        long_signal = False
        short_signal = False

        # 3) 结合 RSI 方向判定（主规则）
        if ed > 0 and 52 <= rsi <= 68:
            long_signal = True
        elif ed < 0 and 32 <= rsi <= 48:
            short_signal = True
        else:
            # 中等趋势 + 极端 RSI 给一次补充机会
            if ed > 0 and rsi > 70:
                long_signal = True
            elif ed < 0 and rsi < 30:
                short_signal = True

        if not (long_signal or short_signal):
            stats.blocked_rsi += 1
            continue

        # === 交易执行 ===
        if position == 0:
            # 开新仓
            position = 1 if long_signal else -1
            position_size = equity * risk_fraction  # 使用当前总资金的一部分
            entry_price = price
            stats.trades += 1
        elif (position == 1 and short_signal) or (position == -1 and long_signal):
            # 反向信号 -> 先平旧仓，再开新仓
            pnl = (price - entry_price) / entry_price * position_size * position
            equity += pnl
            if pnl > 0:
                stats.wins += 1
            # 更新回撤
            peak_equity = max(peak_equity, equity)
            if peak_equity > 0:
                dd = (equity - peak_equity) / peak_equity
                stats.max_drawdown_pct = min(stats.max_drawdown_pct, dd)
            stats.pnl += pnl

            # 开新仓
            position = 1 if long_signal else -1
            position_size = equity * risk_fraction
            entry_price = price
            stats.trades += 1
        else:
            # 同向信号暂时忽略，避免过度交易
            continue

    # 收尾：若有持仓则在最后一根平仓
    if position != 0:
        last_price = float(df["close"].iloc[-1])
        pnl = (last_price - entry_price) / entry_price * position_size * position
        equity += pnl
        if pnl > 0:
            stats.wins += 1
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            dd = (equity - peak_equity) / peak_equity
            stats.max_drawdown_pct = min(stats.max_drawdown_pct, dd)
        stats.pnl += pnl

    return stats, equity


# ===================== 组合回测与汇总 =====================

def run_backtest(
    symbols: List[str],
    days: int,
    seed: int,
    initial_capital: float = 10000.0,
) -> None:
    logger = setup_logger()
    logger.info("🚀 开始回测 ...")

    n = len(symbols)
    capital_per_symbol = initial_capital / n

    all_stats: Dict[str, SymbolStats] = {}
    final_equities: Dict[str, float] = {}
    total_trades = 0
    total_wins = 0

    for idx, sym in enumerate(symbols):
        sym_seed = seed + idx * 1000
        logger.info(f"🔍 测试币种: {sym}")
        df = generate_synthetic_data(sym, days, sym_seed)
        logger.info(f"📊 使用模拟市场数据: {sym} ({len(df)} 行)")
        stats, final_eq = run_backtest_for_symbol(df, sym, capital_per_symbol, logger)
        all_stats[sym] = stats
        final_equities[sym] = final_eq
        total_trades += stats.trades
        total_wins += stats.wins

    total_final = sum(final_equities.values())
    total_pnl = total_final - initial_capital
    avg_winrate = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0

    # 组合最大回撤：取单币种中最差的一个
    if all_stats:
        worst_dd = min(s.max_drawdown_pct for s in all_stats.values())
    else:
        worst_dd = 0.0

    logger.info("=" * 79)
    logger.info("🧠 智能量化交易系统 - 回测报告")
    logger.info("=" * 79)
    logger.info(f"测试币种: {len(symbols)} 个")
    logger.info(f"总交易次数: {total_trades} 笔")
    logger.info(f"总收益: ${total_pnl:.2f} ({total_pnl / initial_capital * 100:.2f}%)")
    logger.info(f"最终资金: ${total_final:.2f}")
    logger.info(f"平均胜率: {avg_winrate:.1f}%")
    logger.info(f"最大回撤(最差单币种): {worst_dd * 100:.1f}%")
    logger.info("")
    logger.info("📊 各币种表现:")

    for sym in symbols:
        st = all_stats.get(sym, SymbolStats())
        winrate = (st.wins / st.trades * 100.0) if st.trades > 0 else 0.0
        logger.info(
            f"  🟡 {sym}: {st.trades} 笔, 胜率: {winrate:.1f}%, "
            f"收益: ${st.pnl:.2f}, 最大回撤: {st.max_drawdown_pct * 100:.1f}%"
        )
        logger.info(
            f"     过滤统计 -> 趋势: {st.blocked_trend}, 波动: {st.blocked_vol}, RSI: {st.blocked_rsi}"
        )

    # 简单风险收益打分（之后可以再升级）
    score = 50.0
    if total_trades > 100:
        score += 5.0
    if total_pnl > 0:
        score += min(10.0, total_pnl / initial_capital * 10)
    score += max(-15.0, worst_dd * 100 * 0.5)  # 回撤越深扣分越多
    score = max(0.0, min(100.0, score))

    logger.info("")
    logger.info(f"🤖 简易风险收益评分: {score:.1f} / 100")


# ===================== CLI =====================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmartBacktest v6.7 (纯模拟版)")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC/USDT,ETH/USDT,SOL/USDT",
        help="用逗号分隔的交易对列表，例如 BTC/USDT,ETH/USDT",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="回测天数（用于生成模拟数据）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，保证重复性",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    run_backtest(symbols, args.days, args.seed)
