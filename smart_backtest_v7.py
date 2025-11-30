#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SmartBacktest v7
==========================
第二季 · 完整升级版（符合你所有要求）

核心升级：
- Step1：真实Binance数据接入（优先）+ fallback 模拟数据
- Step2：多策略合成信号（MACD/EMA/Turtle/BOLL/Breakout）
- 保留趋势过滤 + 波动 + RSI 情绪过滤
- 统一自适应过滤 + 策略库决策框架
"""

import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging

# 你已有的文件
from real_market_data import RealMarketData
from real_strategies import build_ensemble_signal


# ========== 日志设置 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ====================================================
# 模拟市场生成（来自你第一季）
# ====================================================
def generate_mock_data(symbol: str, days: int = 30, seed: Optional[int] = None) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)

    periods = days * 24 * 12  # 5m 一天 288 根

    prices = [100.0]
    for _ in range(periods):
        drift = np.random.normal(0, 1)
        prices.append(prices[-1] * (1 + drift * 0.001))

    df = pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=periods + 1, freq="5min"),
        "open": prices[:-1],
        "high": np.maximum(prices[:-1], prices[1:]),
        "low": np.minimum(prices[:-1], prices[1:]),
        "close": prices[1:],
        "volume": np.random.rand(periods) * 10,
    })

    df.set_index("timestamp", inplace=True)
    return df


# ====================================================
# SymbolResult 结构（与第一季一致）
# ====================================================
class SymbolResult:
    def __init__(self, pnl: float, trades: int, wins: int, max_dd_pct: float):
        self.pnl = pnl
        self.trades = trades
        self.wins = wins
        self.max_dd_pct = max_dd_pct

    @property
    def win_rate(self):
        return (self.wins / self.trades * 100) if self.trades > 0 else 0.0


# ====================================================
# 指标计算（趋势、RSI、波动等）
# ====================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # MA
    d["ma_fast"] = d["close"].rolling(20).mean()
    d["ma_slow"] = d["close"].rolling(50).mean()

    # 趋势方向
    d["trend_long_ok"] = d["ma_fast"] > d["ma_slow"]
    d["trend_short_ok"] = d["ma_fast"] < d["ma_slow"]

    # RSI
    delta = d["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    d["rsi"] = rsi

    d["rsi_long_ok"] = d["rsi"] < 70
    d["rsi_short_ok"] = d["rsi"] > 30

    # ATR
    d["tr"] = np.maximum(
        d["high"] - d["low"],
        np.maximum(abs(d["high"] - d["close"].shift(1)), abs(d["low"] - d["close"].shift(1)))
    )
    d["atr"] = d["tr"].rolling(14).mean()

    return d


# ====================================================
# AdaptiveSignalEngine 核心交易引擎
# ====================================================
class AdaptiveSignalEngine:
    def _build_filters(self, d: pd.DataFrame, symbol: str) -> pd.DataFrame:
        # 保留你的结构（后续可扩展）
        return d

    # ====================================================
    # run_symbol_backtest：已升级版本
    # ====================================================
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

        # Step2：多策略合成信号
        d["strategy_signal"] = build_ensemble_signal(d)

        cash = initial_capital
        equity = initial_capital
        position = 0
        entry_price = 0.0
        size = 0.0

        max_equity = initial_capital

        pnl_total = 0.0
        trades = 0
        wins = 0
        max_dd_pct = 0.0

        for idx, row in d.iterrows():
            price = float(row["close"])
            atr = float(row["atr"]) if not np.isnan(row["atr"]) else 0.0

            # ========= 持仓中：检查止盈止损 =========
            if position != 0:
                # 止损/止盈水平（V7保留简单版）
                sl_price = entry_price * (1 - sl_pct) if position > 0 else entry_price * (1 + sl_pct)
                tp_price = entry_price * (1 + tp_pct) if position > 0 else entry_price * (1 - tp_pct)

                exit_flag = False
                reason = ""

                if position > 0:  # 多单
                    if price <= sl_price:
                        exit_flag = True
                        reason = "SL"
                    elif price >= tp_price:
                        exit_flag = True
                        reason = "TP"
                else:  # 空单
                    if price >= sl_price:
                        exit_flag = True
                        reason = "SL"
                    elif price <= tp_price:
                        exit_flag = True
                        reason = "TP"

                if exit_flag:
                    pnl = (price - entry_price) * size * position
                    pnl_total += pnl
                    cash += pnl
                    trades += 1
                    wins += (pnl > 0)

                    position = 0
                    size = 0
                    continue

            # ========= 空仓状态：寻找入场信号 =========
            if position == 0:
                # 多因子过滤
                trend_long_ok = bool(row["trend_long_ok"] and row["rsi_long_ok"])
                trend_short_ok = bool(row["trend_short_ok"] and row["rsi_short_ok"])

                strat_sig = row["strategy_signal"]

                long_signal = trend_long_ok and (strat_sig > 0)
                short_signal = trend_short_ok and (strat_sig < 0)

                if not (long_signal or short_signal):
                    continue

                # 按固定风险下单
                risk_amount = cash * risk_per_trade
                notional = risk_amount / sl_pct if sl_pct > 0 else 0
                if notional <= 0:
                    continue

                position = 1 if long_signal else -1
                entry_price = price
                size = notional / price

            # ========= 跟踪最大回撤 =========
            equity = cash + (price - entry_price) * size * position if position != 0 else cash
            max_equity = max(max_equity, equity)
            dd_pct = (equity - max_equity) / max_equity * 100
            max_dd_pct = min(max_dd_pct, dd_pct)

        return SymbolResult(
            pnl=pnl_total,
            trades=trades,
            wins=wins,
            max_dd_pct=max_dd_pct,
        )


# ====================================================
# run_backtest（完整升级）
# ====================================================
def run_backtest(
    symbols: List[str],
    days: int,
    initial_capital: float,
    seed: Optional[int] = None,
    data_source: str = "real",  # "real" 或 "mock"
):
    logger.info("🚀 SmartBacktest V7 运行中...")
    logger.info("🪙 币种: %s", symbols)
    logger.info("📅 回测天数: %d", days)
    logger.info("💰 初始资金: %.2f", initial_capital)
    logger.info("📊 数据源: %s", data_source)

    if seed is not None:
        np.random.seed(seed)

    engine = AdaptiveSignalEngine()
    market = RealMarketData()

    symbol_results: Dict[str, SymbolResult] = {}
    total_pnl = 0.0
    total_trades = 0
    total_wins = 0
    worst_dd_pct = 0.0

    capital_per_symbol = initial_capital / len(symbols)

    for sym in symbols:
        logger.info("🔍 处理币种: %s", sym)

        # Step1：真实数据 → fallback 模拟
        if data_source == "real":
            try:
                df = market.get_recent_klines(sym, interval="5m", days=days)
                if df is None or df.empty:
                    logger.warning("⚠️ 真实数据为空，使用模拟市场: %s", sym)
                    df = generate_mock_data(sym, days, seed)
                else:
                    logger.info("📊 使用真实市场数据: %s (%d 行)", sym, len(df))
            except:
                logger.exception("❌ 真实数据获取失败，使用模拟数据")
                df = generate_mock_data(sym, days, seed)
        else:
            df = generate_mock_data(sym, days, seed)

        res = engine.run_symbol_backtest(
            symbol=sym,
            df=df,
            initial_capital=capital_per_symbol,
            max_leverage=3.0,
            risk_per_trade=0.01,
            tp_pct=0.01,
            sl_pct=0.01,
        )

        symbol_results[sym] = res
        total_pnl += res.pnl
        total_trades += res.trades
        total_wins += res.wins
        worst_dd_pct = min(worst_dd_pct, res.max_dd_pct)

    # 汇总报告
    win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0.0

    print("\n========== 🚀 SmartBacktest V7 报告 ==========")
    print(f"总收益: {total_pnl:.2f}")
    print(f"总交易次数: {total_trades}")
    print(f"总胜率: {win_rate:.2f}%")
    print(f"最大回撤: {worst_dd_pct:.2f}%")
    print("\n按币种：")
    for sym, r in symbol_results.items():
        print(f"- {sym} | PnL={r.pnl:.2f}, trades={r.trades}, win_rate={r.win_rate:.2f}%, dd={r.max_dd_pct:.2f}%")

    return symbol_results


# ====================================================
# main（完整版）
# ====================================================
def parse_symbols(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="SmartBacktest V7（第二季升级）")
    parser.add_argument("--symbols", type=str,
                        default="BTC/USDT,ETH/USDT,SOL/USDT")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--initial-capital", type=float, default=10000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-source", type=str,
                        choices=["real", "mock"], default="real")

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
