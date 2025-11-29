#!/usr/bin/env python3
"""
high_frequency_backtest.py

高频交易回测系统 - 简化可工作版本 + 结果分析/打分
适合用来做：
- 信号管线是否能跑通；
- 各币种大致表现的对比；
- 给 AI/人类一个大概的“好坏点评”。

注意：
真正的实战策略评估（Sharpe / 回撤 / 因子分析）建议放到 smart_backtest 里做。
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any

import warnings

warnings.filterwarnings("ignore")

# 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("HighFrequencyBacktest")


class SimpleSignalDetector:
    """简化信号检测器 - 只保证“有信号，可回测”"""

    def analyze_enhanced_signals(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        返回一个 DataFrame:
        - signal_strength: [-1, 1]
        - signal_type: STRONG_BUY/BUY/SELL/STRONG_SELL/HOLD
        """
        try:
            if data is None or len(data) < 20 or "close" not in data.columns:
                return pd.DataFrame()

            signals = []
            closes = data["close"].values

            for i in range(len(data)):
                if i < 20:
                    signals.append({"signal_strength": 0.0, "signal_type": "HOLD"})
                    continue

                current_price = closes[i]
                sma_short = np.mean(closes[i - 5 : i])
                sma_long = np.mean(closes[i - 20 : i])

                if sma_short > sma_long and current_price > sma_short:
                    strength = 0.8
                    stype = "STRONG_BUY"
                elif sma_short < sma_long and current_price < sma_short:
                    strength = -0.8
                    stype = "STRONG_SELL"
                elif sma_short > sma_long:
                    strength = 0.3
                    stype = "BUY"
                elif sma_short < sma_long:
                    strength = -0.3
                    stype = "SELL"
                else:
                    strength = 0.0
                    stype = "HOLD"

                signals.append({"signal_strength": strength, "signal_type": stype})

            return pd.DataFrame(signals)

        except Exception as e:
            logger.error(f"信号分析错误 [{symbol}]: {e}")
            return pd.DataFrame()


class HighFrequencyBacktest:
    """高频交易回测系统（简化版）"""

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        compound_mode: bool = True,
        leverage: float = 3.0,
        signal_detector: SimpleSignalDetector | None = None,
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.compound_mode = compound_mode
        self.leverage = leverage

        # 这里保留为对象属性，但每次单币种回测内部会自己维护局部 state
        self.positions: Dict[str, dict] = {}
        self.trade_history: List[dict] = []

        self.signal_detector = signal_detector or SimpleSignalDetector()

        # “合理价格”区间只是为了防止生成出离谱数据（调试用）
        self.reasonable_price_ranges = {
            "BTC/USDT": (15_000, 80_000),
            "ETH/USDT": (800, 5_000),
            "SOL/USDT": (10, 300),
            "BNB/USDT": (100, 800),
            "ADA/USDT": (0.2, 3),
            "DOT/USDT": (2, 50),
            "AVAX/USDT": (5, 100),
            "LINK/USDT": (3, 50),
            "MATIC/USDT": (0.3, 3),
        }

        logger.info("🚀 高频回测系统初始化完成 - 简化可工作版本")

    # ------------------------------------------------------------------ #
    # 数据生成（如果你有真实数据，可以自行替换成读取 CSV / API）
    # ------------------------------------------------------------------ #
    def _generate_sample_data(self, symbol: str, days: int) -> pd.DataFrame:
        """生成简单但不离谱的模拟 OHLCV 数据"""
        dates = pd.date_range(end=datetime.now(), periods=days * 24, freq="H")

        base_prices = {
            "BTC/USDT": 35_000,
            "ETH/USDT": 2_500,
            "SOL/USDT": 100,
            "BNB/USDT": 300,
            "ADA/USDT": 0.5,
            "DOT/USDT": 6,
            "AVAX/USDT": 20,
            "LINK/USDT": 15,
            "MATIC/USDT": 0.8,
        }

        base_price = base_prices.get(symbol, 100.0)

        np.random.seed(42)
        # 简单随机游走：日化波动控制在合理范围
        returns = np.random.normal(0.0002, 0.015, len(dates))
        prices = base_price * (1 + returns).cumprod()

        data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices * 0.998,
                "high": prices * 1.005,
                "low": prices * 0.995,
                "close": prices,
                "volume": np.random.uniform(10_000, 500_000, len(dates)),
            }
        )

        return data

    # ------------------------------------------------------------------ #
    # 回测主流程
    # ------------------------------------------------------------------ #
    def run_backtest(
        self, symbols: List[str], days: int = 30, test_full_year: bool = False
    ) -> List[Dict[str, Any]]:
        """运行多币种回测 - 返回每个币种的统计结果"""
        logger.info(f"🎯 开始回测: {symbols}，天数={days}")

        all_results: List[Dict[str, Any]] = []

        for symbol in symbols:
            logger.info(f"\n🔍 测试币种: {symbol}")

            try:
                data = self._generate_sample_data(symbol, days)
                logger.info(f"✅ 生成 {symbol} 模拟数据: {len(data)} 条")

                result = self._backtest_single_symbol(symbol, data)
                all_results.append(result)

            except Exception as e:
                logger.error(f"❌ {symbol} 回测失败: {e}")
                continue

        self._generate_report(all_results)
        self._analyze_results(all_results)

        return all_results

    def _backtest_single_symbol(
        self, symbol: str, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """单币种回测（局部 state，避免串币种污染）"""
        symbol_positions: Dict[str, dict] = {}
        symbol_trades: List[dict] = []

        current_capital = self.initial_capital

        for i in range(50, len(data)):
            try:
                row = data.iloc[i]
                current_price = float(row["close"])
                current_time = row["timestamp"]

                signal_data = data.iloc[: i + 1]
                signals = self.signal_detector.analyze_enhanced_signals(
                    signal_data, symbol
                )

                if signals.empty or i >= len(signals):
                    continue

                signal_row = signals.iloc[i]
                signal_strength = float(signal_row.get("signal_strength", 0.0))

                trade_result, current_capital = self._execute_trading_logic(
                    symbol,
                    current_price,
                    current_time,
                    signal_strength,
                    symbol_positions,
                    current_capital,
                )

                if trade_result:
                    symbol_trades.append(trade_result)

            except Exception as e:
                logger.error(f"❌ {symbol} 回测迭代错误: {e}")
                continue

        # 统计结果
        total_pnl = sum(t.get("pnl", 0.0) for t in symbol_trades)
        metrics = self._evaluate_symbol_trades(
            symbol_trades, initial_capital=self.initial_capital
        )

        return {
            "symbol": symbol,
            "trades": symbol_trades,
            "total_trades": len(symbol_trades),
            "total_pnl": total_pnl,
            "metrics": metrics,
        }

    def _execute_trading_logic(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        signal_strength: float,
        positions: Dict[str, dict],
        current_capital: float,
    ):
        """
        执行简单交易逻辑：
        - signal_strength > 0.7 开多
        - signal_strength < -0.7 开空
        - 已有仓位则按“持有时间或信号反转”平仓
        """
        trade = None

        try:
            # 开仓逻辑
            if signal_strength > 0.7 and symbol not in positions:
                position_size = current_capital * 0.1
                positions[symbol] = {
                    "type": "long",
                    "entry_price": price,
                    "size": position_size,
                    "timestamp": timestamp,
                }
                trade = {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "action": "BUY",
                    "price": price,
                    "size": position_size,
                    "type": "long",
                }

            elif signal_strength < -0.7 and symbol not in positions:
                position_size = current_capital * 0.1
                positions[symbol] = {
                    "type": "short",
                    "entry_price": price,
                    "size": position_size,
                    "timestamp": timestamp,
                }
                trade = {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "action": "SELL",
                    "price": price,
                    "size": position_size,
                    "type": "short",
                }

            # 平仓逻辑
            elif symbol in positions:
                position = positions[symbol]
                hold_hours = (timestamp - position["timestamp"]).total_seconds() / 3600

                should_close = False
                if position["type"] == "long" and (
                    hold_hours > 12 or signal_strength < -0.3
                ):
                    should_close = True
                elif position["type"] == "short" and (
                    hold_hours > 12 or signal_strength > 0.3
                ):
                    should_close = True

                if should_close:
                    if position["type"] == "long":
                        pnl = (
                            (price - position["entry_price"])
                            / position["entry_price"]
                            * position["size"]
                            * self.leverage
                        )
                    else:
                        pnl = (
                            (position["entry_price"] - price)
                            / position["entry_price"]
                            * position["size"]
                            * self.leverage
                        )

                    trade = {
                        "symbol": symbol,
                        "timestamp": timestamp,
                        "action": "CLOSE",
                        "price": price,
                        "pnl": pnl,
                        "type": position["type"],
                        "hold_hours": hold_hours,
                    }

                    current_capital += pnl
                    del positions[symbol]

        except Exception as e:
            logger.error(f"交易执行错误 {symbol}: {e}")

        return trade, current_capital

    # ------------------------------------------------------------------ #
    # 统计 & 报告
    # ------------------------------------------------------------------ #
    def _evaluate_symbol_trades(
        self, trades: List[dict], initial_capital: float
    ) -> Dict[str, float]:
        """对单个币种的交易结果做个简单评价"""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_profit": 0.0,
            }

        pnls = [t.get("pnl", 0.0) for t in trades if "pnl" in t]
        total_pnl = sum(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls) * 100 if pnls else 0.0
        avg_profit = total_pnl / len(pnls) if pnls else 0.0

        # 简单 profit_factor
        loss_sum = abs(sum(losses)) if losses else 0.0
        profit_sum = sum(wins) if wins else 0.0
        profit_factor = profit_sum / loss_sum if loss_sum > 0 else float("inf")

        # 简单“收益率”（总 PnL / 初始资金）
        total_return = total_pnl / initial_capital if initial_capital > 0 else 0.0

        return {
            "total_trades": len(pnls),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_profit": avg_profit,
            "profit_factor": profit_factor,
            "total_return": total_return,
        }

    def _generate_report(self, all_results: List[Dict[str, Any]]):
        """打印一份简单汇总表"""
        logger.info("\n" + "=" * 80)
        logger.info("🎯 高频交易系统 - 回测报告 (简化可工作版本)")
        logger.info("=" * 80)

        logger.info("\n📊 币种表现统计:")
        logger.info("币种          交易数    胜率     总收益      平均收益   盈亏比")
        logger.info("-" * 80)

        total_trades_all = 0
        total_pnl_all = 0.0
        win_rates = []

        for result in all_results:
            symbol = result["symbol"]
            metrics = result["metrics"]
            trades = metrics.get("total_trades", 0)
            win_rate = metrics.get("win_rate", 0.0)
            total_pnl = metrics.get("total_pnl", 0.0)
            avg_profit = metrics.get("avg_profit", 0.0)
            profit_factor = metrics.get("profit_factor", 0.0)

            logger.info(
                f"{symbol:12} {trades:6d}   {win_rate:5.1f}%   "
                f"${total_pnl:8.2f}   ${avg_profit:8.2f}   {profit_factor:5.2f}"
            )

            if trades > 0:
                total_trades_all += trades
                total_pnl_all += total_pnl
                win_rates.append(win_rate)

        if total_trades_all > 0:
            avg_win_rate = float(np.mean(win_rates)) if win_rates else 0.0
            logger.info("-" * 80)
            logger.info(f"📈 总交易次数: {total_trades_all}")
            logger.info(f"📈 平均胜率: {avg_win_rate:.1f}%")
            logger.info(f"💰 总收益: ${total_pnl_all:+.2f}")
            logger.info(
                f"💰 平均每笔收益: ${total_pnl_all / total_trades_all:+.2f}"
            )
        else:
            logger.info("❌ 没有产生任何交易")

    def _analyze_results(self, all_results: List[Dict[str, Any]]):
        """
        “AI 风格”点评一下结果（只是规则逻辑，但方便你一眼看出问题）
        """
        logger.info("\n🧠 AI-style 结果分析:")

        if not all_results:
            logger.info("  没有任何结果，先检查数据或信号生成。")
            return

        # 找出收益最好的 / 最差的币种
        valid = [r for r in all_results if r["metrics"]["total_trades"] > 0]
        if not valid:
            logger.info("  所有币种都没有交易，说明信号太严格或逻辑有问题。")
            return

        best = max(valid, key=lambda r: r["metrics"]["total_pnl"])
        worst = min(valid, key=lambda r: r["metrics"]["total_pnl"])

        logger.info(
            f"  ✅ 表现最佳: {best['symbol']} | PnL={best['metrics']['total_pnl']:.2f}, "
            f"WinRate={best['metrics']['win_rate']:.1f}%"
        )
        logger.info(
            f"  ❌ 表现最差: {worst['symbol']} | PnL={worst['metrics']['total_pnl']:.2f}, "
            f"WinRate={worst['metrics']['win_rate']:.1f}%"
        )

        # 简单建议
        for r in valid:
            symbol = r["symbol"]
            m = r["metrics"]
            if m["win_rate"] < 40 and m["profit_factor"] < 1.0:
                logger.info(
                    f"  💡 {symbol}: 胜率<40% 且 盈亏比<1，建议："
                    f"减少交易频率/提高开仓阈值，或在 smart_backtest 中直接淘汰该策略组合。"
                )
            elif m["win_rate"] > 55 and m["profit_factor"] > 1.5:
                logger.info(
                    f"  🌟 {symbol}: 胜率 & 盈亏比都不错，可以在 smart_backtest 里重点精调参数。"
                )


def main():
    parser = argparse.ArgumentParser(description="高频交易回测系统 - 简化可工作版本")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC/USDT,ETH/USDT,SOL/USDT",
        help="交易对，用逗号分隔",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="回测天数",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10_000,
        help="初始资金",
    )

    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",")]

    backtest = HighFrequencyBacktest(initial_capital=args.capital)
    backtest.run_backtest(symbols=symbols, days=args.days)


if __name__ == "__main__":
    main()
