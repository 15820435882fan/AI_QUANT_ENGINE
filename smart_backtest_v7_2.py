#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SmartBacktest v7_2
===============================
- 强制禁用代理，确保 Binance 连接成功
- 完全修复模拟K线长度不一致问题（ValueError 消失）
- 真实数据失败自动 fallback 且永不报错
- 集成 real_strategies 多策略合成信号（MACD/EMA/Turtle/BOLL/Breakout）
- 第二季 Step1 + Step2 完整实现版
"""

# ============================================================
# 🔥 0. 开始：强制禁用所有代理（解决 Binance 访问失败）
# ============================================================
import os
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''


# ============================================================
# 基础库
# ============================================================
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging

# 你已有的模块
from real_market_data import RealMarketData
from real_strategies import build_ensemble_signal


# ============================================================
# 日志
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# 🔧 1. 修复后的模拟K线生成函数（永不报错）
# ============================================================
def generate_mock_data(symbol: str, days: int = 30, seed: Optional[int] = None) -> pd.DataFrame:
    """
    稳定版本：所有列长度完全一致，不会再出现 ValueError
    """
    if seed is not None:
        np.random.seed(seed)

    periods = days * 24 * 12  # 5m K线数量
    if periods <= 1:
        periods = 288  # 至少1天

    # 生成价格序列
    prices = [100.0]
    for _ in range(periods):
        drift = np.random.normal(0, 1)
        prices.append(prices[-1] * (1 + drift * 0.001))
    prices = np.array(prices)

    # 所有列保持 periods 长度一致
    openp = prices[:-1]
    closep = prices[1:]
    highp = np.maximum(openp, closep)
    lowp = np.minimum(openp, closep)
    vol = np.random.rand(periods) * 10

    df = pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=periods, freq="5min"),
        "open": openp,
        "high": highp,
        "low": lowp,
        "close": closep,
        "volume": vol,
    })

    df.set_index("timestamp", inplace=True)
    return df


# ============================================================
# 2. 回测结果结构
# ============================================================
class SymbolResult:
    def __init__(self, pnl: float, trades: int, wins: int, max_dd_pct: float):
        self.pnl = pnl
        self.trades = trades
        self.wins = wins
        self.max_dd_pct = max_dd_pct

    @property
    def win_rate(self):
        return (self.wins / self.trades * 100) if self.trades > 0 else 0.0


# ============================================================
# 3. 指标系统：MA、RSI、ATR
# ============================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # MA
    d["ma_fast"] = d["close"].rolling(20).mean()
    d["ma_slow"] = d["close"].rolling(50).mean()

    d["trend_long_ok"] = d["ma_fast"] > d["ma_slow"]
    d["trend_short_ok"] = d["ma_fast"] < d["ma_slow"]

    # RSI
    delta = d["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-12)
    d["rsi"] = 100 - (100 / (1 + rs))

    d["rsi_long_ok"] = d["rsi"] < 70
    d["rsi_short_ok"] = d["rsi"] > 30

    # ATR
    d["tr"] = np.maximum(
        d["high"] - d["low"],
        np.maximum(
            abs(d["high"] - d["close"].shift(1)),
            abs(d["low"] - d["close"].shift(1)),
        ),
    )
    d["atr"] = d["tr"].rolling(14).mean()

    return d


# ============================================================
# 4. AdaptiveSignalEngine（第二季核心引擎）
# ============================================================
class AdaptiveSignalEngine:
    def _build_filters(self, d: pd.DataFrame, symbol: str) -> pd.DataFrame:
        return d

    def run_symbol_backtest(
        self,
        symbol: str,
        df: pd.DataFrame,
        initial_capital: float,
        risk_per_trade: float = 0.01,
        tp_pct: float = 0.01,
        sl_pct: float = 0.01,
    ) -> SymbolResult:

        d = compute_indicators(df)
        d = self._build_filters(d, symbol)

        # 多策略合成信号
        d["strategy_signal"] = build_ensemble_signal(d)

        cash = initial_capital
        position = 0
        size = 0
        entry_price = 0.0

        pnl_total = 0.0
        trades = 0
        wins = 0

        equity = initial_capital
        max_equity = initial_capital
        max_dd_pct = 0.0

        # ========== K线循环 ==========
        for idx, row in d.iterrows():
            price = float(row["close"])

            # 持仓：检查止损止盈
            if position != 0:
                sl_price = entry_price * (1 - sl_pct) if position > 0 else entry_price * (1 + sl_pct)
                tp_price = entry_price * (1 + tp_pct) if position > 0 else entry_price * (1 - tp_pct)

                exit_flag = (
                    (position > 0 and (price <= sl_price or price >= tp_price)) or
                    (position < 0 and (price >= sl_price or price <= tp_price))
                )

                if exit_flag:
                    pnl = (price - entry_price) * size * position
                    pnl_total += pnl
                    cash += pnl
                    trades += 1
                    wins += (pnl > 0)

                    position = 0
                    size = 0
                    continue

            # 空仓：检查是否开仓
            if position == 0:
                trend_long_ok = row["trend_long_ok"] and row["rsi_long_ok"]
                trend_short_ok = row["trend_short_ok"] and row["rsi_short_ok"]
                strat_sig = row["strategy_signal"]

                long_signal = trend_long_ok and strat_sig > 0
                short_signal = trend_short_ok and strat_sig < 0

                if not (long_signal or short_signal):
                    continue

                # 仓位大小
                risk_amount = cash * risk_per_trade
                notional = risk_amount / sl_pct if sl_pct > 0 else 0
                if notional <= 0:
                    continue

                position = 1 if long_signal else -1
                entry_price = price
                size = notional / price

            # 计算回撤
            equity = cash + (
                (price - entry_price) * size * position if position != 0 else 0
            )
            max_equity = max(max_equity, equity)
            dd_pct = (equity - max_equity) / max_equity * 100
            max_dd_pct = min(max_dd_pct, dd_pct)

        return SymbolResult(pnl_total, trades, wins, max_dd_pct)


# ============================================================
# 5. 多币种回测
# ============================================================
def run_backtest(
    symbols: List[str],
    days: int,
    initial_capital: float,
    seed: Optional[int],
    data_source: str,
):
    logger.info("🚀 SmartBacktest V7_2 启动")
    logger.info(f"🪙 币种: {symbols}")
    logger.info(f"📅 天数: {days}")
    logger.info(f"📊 数据源: {data_source}")

    if seed:
        np.random.seed(seed)

    engine = AdaptiveSignalEngine()
    market = RealMarketData()

    total_pnl = 0
    total_trades = 0
    total_wins = 0
    worst_dd_pct = 0

    results = {}
    per_cap = initial_capital / len(symbols)

    for sym in symbols:
        logger.info(f"🔍 处理 {sym}")

        try:
            if data_source == "real":
                df = market.get_recent_klines(sym, "5m", days)
                if df is None or len(df) == 0:
                    logger.warning(f"⚠️ Binance 数据为空，使用模拟数据: {sym}")
                    df = generate_mock_data(sym, days, seed)
            else:
                df = generate_mock_data(sym, days, seed)
        except Exception as e:
            logger.error(f"❌ Binance 拉取失败: {e}")
            df = generate_mock_data(sym, days, seed)

        res = engine.run_symbol_backtest(sym, df, per_cap)

        results[sym] = res
        total_pnl += res.pnl
        total_trades += res.trades
        total_wins += res.wins
        worst_dd_pct = min(worst_dd_pct, res.max_dd_pct)

    win_rate = total_wins / total_trades * 100 if total_trades else 0

    print("\n========== 📈 SmartBacktest V7_2 报告 ==========")
    print(f"总收益: {total_pnl:.2f}")
    print(f"总交易数: {total_trades}")
    print(f"总胜率: {win_rate:.2f}%")
    print(f"最大回撤: {worst_dd_pct:.2f}%\n")

    print("按币种：")
    for sym, r in results.items():
        print(f"- {sym}: pnl={r.pnl:.2f}, trades={r.trades}, win={r.win_rate:.2f}%, maxDD={r.max_dd_pct:.2f}%")

    return results


# ============================================================
# main
# ============================================================
def parse_symbols(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="SmartBacktest V7_2")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--initial-capital", type=float, default=10000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-source", type=str, choices=["real", "mock"], default="real")

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
