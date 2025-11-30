import argparse
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from real_market_data import RealMarketData
from real_strategies import build_ensemble_signal
import numpy as np
import pandas as pd


logger = logging.getLogger("SmartBacktest")


# =========================
# 基础工具
# =========================

def setup_logger():
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def parse_symbols(symbols_str: str) -> List[str]:
    return [s.strip() for s in symbols_str.split(",") if s.strip()]


# =========================
# 市场数据（模拟版）
# =========================

def generate_mock_data(symbol: str, days: int, seed: Optional[int] = None) -> pd.DataFrame:
    if seed is not None:
        random.seed(seed + hash(symbol) % 1000)
        np.random.seed(seed + hash(symbol) % 1000)

    n = days * 288
    dt_index = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="5min")

    base_price = {
        "BTC/USDT": 50000,
        "ETH/USDT": 3500,
        "SOL/USDT": 150,
    }.get(symbol, 100)

    vol_scale = {
        "BTC/USDT": 0.015,
        "ETH/USDT": 0.02,
        "SOL/USDT": 0.03,
    }.get(symbol, 0.02)

    prices = [base_price]
    for _ in range(n - 1):
        drift = random.uniform(-vol_scale, vol_scale)
        revert = (base_price - prices[-1]) * 0.0005
        new_price = max(0.1, prices[-1] * (1 + drift) + revert)
        prices.append(new_price)

    prices = np.array(prices)

    close = prices
    open_ = close * (1 + np.random.normal(0, 0.001, size=n))
    high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.0015, size=n)))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.0015, size=n)))
    volume = np.abs(np.random.normal(1, 0.1, size=n)) * base_price * 0.001

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dt_index,
    )
    return df


# =========================
# 指标 & 自适应过滤
# =========================

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean()
    df["atr_pct"] = atr / close.replace(0, np.nan)

    ema_fast = close.ewm(span=20, adjust=False).mean()
    ema_slow = close.ewm(span=60, adjust=False).mean()
    df["trend_raw"] = (ema_fast - ema_slow) / close.replace(0, np.nan)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

    df = df.dropna().copy()
    return df


@dataclass
class FilterStats:
    trend_filter_pass: int = 0
    vol_filter_pass: int = 0
    rsi_filter_pass: int = 0
    total_bars: int = 0


@dataclass
class SymbolResult:
    trades: int = 0
    wins: int = 0
    pnl: float = 0.0
    max_dd_pct: float = 0.0
    filter_stats: FilterStats = field(default_factory=FilterStats)


class AdaptiveSignalEngine:
    def __init__(self):
        self.filter_stats: Dict[str, FilterStats] = {}

    def _build_filters(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        d = df.copy()
        stats = FilterStats()

        atr_pct = d["atr_pct"].clip(lower=0)
        stats.total_bars = len(d)

        low_q = atr_pct.quantile(0.25)
        high_q = atr_pct.quantile(0.90)
        d["vol_ok"] = (atr_pct >= low_q) & (atr_pct <= high_q)
        stats.vol_filter_pass = int(d["vol_ok"].sum())

        trend_abs = d["trend_raw"].abs()
        trend_thr = trend_abs.quantile(0.6)
        d["trend_long_ok"] = d["trend_raw"] > trend_thr
        d["trend_short_ok"] = d["trend_raw"] < -trend_thr
        stats.trend_filter_pass = int(d["trend_long_ok"].sum() + d["trend_short_ok"].sum())

        rsi = d["rsi"].clip(lower=0, upper=100)
        long_low = rsi.quantile(0.45)
        long_high = rsi.quantile(0.75)
        short_low = rsi.quantile(0.25)
        short_high = rsi.quantile(0.55)

        d["rsi_long_ok"] = (rsi >= long_low) & (rsi <= long_high)
        d["rsi_short_ok"] = (rsi >= short_low) & (rsi <= short_high)
        stats.rsi_filter_pass = int(d["rsi_long_ok"].sum() + d["rsi_short_ok"].sum())

        self.filter_stats[symbol] = stats
        return d

    def run_symbol_backtest(
        self,
        symbol: str,
        df: pd.DataFrame,
        initial_capital: float,
        max_leverage: float = 3.0,
        risk_per_trade: float = 0.01,
        tp_pct: float = 0.01,
        sl_pct: float = 0.01,
    ) -> SymbolResult:
        d = compute_indicators(df)
        d = self._build_filters(d, symbol)

        cash = initial_capital
        equity_peak = cash
        max_dd_pct = 0.0

        position = 0  
        entry_price = 0.0
        position_size = 0.0

        trades = 0
        wins = 0

        for idx, row in d.iterrows():
            price = float(row["close"])

            equity = cash
            if position != 0 and position_size != 0:
                pos_value = position_size * price
                equity += pos_value - (position_size * entry_price)

            equity_peak = max(equity_peak, equity)
            if equity_peak > 0:
                dd_pct = (equity - equity_peak) / equity_peak
                max_dd_pct = min(max_dd_pct, dd_pct)

            if position == 0:
                if not row["vol_ok"]:
                    continue

                risk_amount = cash * risk_per_trade
                if risk_amount <= 0:
                    continue

                notional = (risk_amount / sl_pct) if sl_pct > 0 else 0
                notional = min(notional, cash * max_leverage)

                if notional <= 0:
                    continue

                long_signal = bool(row["trend_long_ok"] and row["rsi_long_ok"])
                short_signal = bool(row["trend_short_ok"] and row["rsi_short_ok"])

                if not (long_signal or short_signal):
                    continue

                position = 1 if long_signal else -1
                entry_price = price
                position_size = notional / price
                trades += 1
                continue

            if position != 0 and position_size > 0:
                pnl_pct = (price / entry_price - 1.0) * position

                exit_reason = None
                if pnl_pct >= tp_pct:
                    exit_reason = "TP"
                elif pnl_pct <= -sl_pct:
                    exit_reason = "SL"
                elif not row["vol_ok"]:
                    exit_reason = "VOL_EXIT"

                if exit_reason:
                    cash += position_size * (price - entry_price) * position
                    if pnl_pct > 0:
                        wins += 1

                    position = 0
                    entry_price = 0.0
                    position_size = 0.0

        final_equity = cash
        if position != 0 and position_size > 0:
            final_equity += position_size * (df["close"].iloc[-1] - entry_price) * position

        pnl = final_equity - initial_capital

        return SymbolResult(
            trades=trades,
            wins=wins,
            pnl=pnl,
            max_dd_pct=max_dd_pct * 100,
            filter_stats=self.filter_stats.get(symbol, FilterStats()),
        )


# =========================
# 顶层回测控制
# =========================

def run_backtest(
    symbols: List[str],
    days: int,
    initial_capital: float,
    seed: Optional[int] = None,
    data_source: str = "real",  # "real" 或 "mock"
):
    """
    SmartBacktest v6.8.1 升级版：
    - 支持真实 Binance K 线（优先）
    - 支持模拟 K 线作为 fallback 或强制模式
    """
    logger.info("🚀 SmartBacktest v6.8.1 运行中...")
    logger.info("🪙 币种: %s", symbols)
    logger.info("📅 回测天数: %d", days)
    logger.info("💰 初始资金: %.2f", initial_capital)
    logger.info("📊 数据源: %s", data_source)

    if seed is not None:
        np.random.seed(seed)

    engine = AdaptiveSignalEngine()
    market_data = RealMarketData()  # ✅ 新增：真实市场数据接口

    symbol_results: Dict[str, SymbolResult] = {}
    total_trades = 0
    total_wins = 0
    total_pnl = 0.0
    worst_dd_pct = 0.0

    capital_per_symbol = initial_capital / max(1, len(symbols))

    for sym in symbols:
        logger.info("🔍 测试币种: %s", sym)

        # -----------------------------
        # Step1: 接入真实 Binance 数据
        # -----------------------------
        if data_source == "real":
            try:
                # 这里假设 RealMarketData 有 get_recent_klines 方法
                df = market_data.get_recent_klines(
                    symbol=sym,
                    interval="5m",
                    days=days,
                )
                if df is not None and not df.empty:
                    logger.info("📊 使用真实市场数据: %s (%d 行)", sym, len(df))
                else:
                    logger.warning("⚠️ 真实数据为空，回退到模拟数据: %s", sym)
                    df = generate_mock_data(sym, days=days, seed=seed)
                    logger.info("📊 使用模拟市场数据: %s (%d 行)", sym, len(df))
            except Exception as e:
                logger.exception("❌ 获取真实数据失败，将使用模拟数据: %s", e)
                df = generate_mock_data(sym, days=days, seed=seed)
                logger.info("📊 使用模拟市场数据: %s (%d 行)", sym, len(df))
        else:
            # 强制使用模拟数据
            df = generate_mock_data(sym, days=days, seed=seed)
            logger.info("📊 使用模拟市场数据: %s (%d 行)", sym, len(df))

        if df is None or df.empty:
            logger.error("❌ %s 数据为空，跳过该币种", sym)
            continue

        res = engine.run_symbol_backtest(
            sym,
            df,
            initial_capital=capital_per_symbol,
            max_leverage=3.0,
            risk_per_trade=0.01,
            tp_pct=0.01,
            sl_pct=0.01,
        )

        symbol_results[sym] = res
        total_trades += res.trades
        total_wins += res.wins
        total_pnl += res.pnl
        worst_dd_pct = min(worst_dd_pct, res.max_dd_pct)

    # ========= 汇总报告 =========
    if total_trades > 0:
        win_rate = total_wins / total_trades * 100
    else:
        win_rate = 0.0

    logger.info("========== 总体回测结果 ==========")
    logger.info("总收益: %.2f", total_pnl)
    logger.info("总交易次数: %d", total_trades)
    logger.info("总胜率: %.2f%%", win_rate)
    logger.info("最大回撤: %.2f%%", worst_dd_pct)

    print("\n===== 汇总报告 =====")
    print(f"总收益: {total_pnl:.2f}")
    print(f"总交易次数: {total_trades}")
    print(f"总胜率: {win_rate:.2f}%")
    print(f"最大回撤: {worst_dd_pct:.2f}%")
    print("\n按币种统计:")
    for sym, r in symbol_results.items():
        print(f"- {sym}: PnL={r.pnl:.2f}, trades={r.trades}, win_rate={r.win_rate:.2f}%, maxDD={r.max_dd_pct:.2f}%")

    return symbol_results



def main():
    parser = argparse.ArgumentParser(description="SmartBacktest v6.8.1 - 自适应过滤版（第二季升级）")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC/USDT,ETH/USDT,SOL/USDT",
        help="逗号分隔的交易对，例如: BTC/USDT,ETH/USDT",
    )
    parser.add_argument("--days", type=int, default=30, help="回测天数")
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="初始资金")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可选）")
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["real", "mock"],
        default="real",
        help="数据源: real=优先真实Binance数据, mock=纯模拟K线",
    )
    args = parser.parse_args()

    symbols = parse_symbols(args.symbols)

    run_backtest(
        symbols=symbols,
        days=args.days,
        initial_capital=args.initial_capital,
        seed=args.seed,
        data_source=args.data_source,
    )



if __name__ == "__main__":
    main()
