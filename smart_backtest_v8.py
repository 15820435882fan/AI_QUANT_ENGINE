#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SmartBacktest v8
===============================
第二季 · Step3：风险与盈利模型强化版

特性：
- 真实 Binance 数据 + fallback 模拟
- 多策略合成信号（来自 real_strategies）
- ATR 驱动的止损与止盈（非固定百分比）
- RR ≥ 1.5（默认为 2.0）
- ATR 移动止损（Trailing Stop）
- 连续亏损冷静期（熄火保护）
"""

# ============================================================
# 0. 强制禁用代理，避免 Binance 被代理劫持
# ============================================================
import os
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

# ============================================================
# 基础库
# ============================================================
import argparse
import logging
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from real_market_data import RealMarketData
from real_strategies import build_ensemble_signal


# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. 模拟 K 线生成（稳定版）
# ============================================================
def generate_mock_data(symbol: str, days: int = 30, seed: Optional[int] = None) -> pd.DataFrame:
    """
    生成一个简易的随机 5m K 线，用于没有真实数据时的 fallback。
    """
    if seed is not None:
        np.random.seed(seed)

    periods = days * 24 * 12  # 5 分钟 K 线数量
    if periods <= 1:
        periods = 288  # 至少 1 天

    prices = [100.0]
    for _ in range(periods):
        drift = np.random.normal(0, 1)
        prices.append(prices[-1] * (1 + drift * 0.001))
    prices = np.array(prices)

    openp = prices[:-1]
    closep = prices[1:]
    highp = np.maximum(openp, closep)
    lowp = np.minimum(openp, closep)
    vol = np.random.rand(periods) * 10

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                end=pd.Timestamp.now(), periods=periods, freq="5min"
            ),
            "open": openp,
            "high": highp,
            "low": lowp,
            "close": closep,
            "volume": vol,
        }
    )
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
    def win_rate(self) -> float:
        return (self.wins / self.trades * 100) if self.trades > 0 else 0.0


# ============================================================
# 3. 指标计算：MA、RSI、ATR
# ============================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # 均线
    d["ma_fast"] = d["close"].rolling(20).mean()
    d["ma_slow"] = d["close"].rolling(50).mean()

    d["trend_long_ok"] = d["ma_fast"] > d["ma_slow"]
    d["trend_short_ok"] = d["ma_fast"] < d["ma_slow"]

    # RSI
    delta = d["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    d["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    d["rsi_long_ok"] = d["rsi"] < 70
    d["rsi_short_ok"] = d["rsi"] > 30

    # ATR
    high_low = d["high"] - d["low"]
    high_close = (d["high"] - d["close"].shift(1)).abs()
    low_close = (d["low"] - d["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    d["tr"] = tr
    d["atr"] = d["tr"].rolling(14).mean()

    return d


# ============================================================
# 4. 自适应信号引擎（V8）
# ============================================================
class AdaptiveSignalEngine:
    """
    V8 核心引擎：
    - 多因子过滤（趋势 + RSI）
    - 多策略合成信号（来自 real_strategies.build_ensemble_signal）
    - ATR 止损 / 止盈 + Trailing Stop
    - 连续亏损冷静期
    """

    def __init__(
        self,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 3.0,
        trail_atr_mult: float = 1.5,
        min_rr: float = 1.5,
        risk_per_trade: float = 0.01,
        max_loss_streak: int = 3,
        cooldown_bars: int = 12 * 12,  # 12小时冷静期（5mK）
    ):
        # 风控参数
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.trail_atr_mult = trail_atr_mult
        self.min_rr = min_rr
        self.risk_per_trade = risk_per_trade

        # 连续亏损控制
        self.max_loss_streak = max_loss_streak
        self.cooldown_bars = cooldown_bars

    def _build_filters(self, d: pd.DataFrame, symbol: str) -> pd.DataFrame:
        # 这里预留位置做更复杂的多周期过滤
        return d

    def run_symbol_backtest(
        self,
        symbol: str,
        df: pd.DataFrame,
        initial_capital: float,
    ) -> SymbolResult:
        d = compute_indicators(df)
        d = self._build_filters(d, symbol)

        # 多策略合成信号（+1 / 0 / -1）
        d["strategy_signal"] = build_ensemble_signal(d)

        cash = initial_capital
        position = 0  # 0=空仓, 1=多, -1=空
        size = 0.0
        entry_price = 0.0
        sl_price = 0.0
        tp_price = 0.0

        pnl_total = 0.0
        trades = 0
        wins = 0

        equity = initial_capital
        max_equity = initial_capital
        max_dd_pct = 0.0

        loss_streak = 0
        cooldown_left = 0  # 冷静期剩余bar数

        for idx, row in d.iterrows():
            price = float(row["close"])
            atr = float(row["atr"]) if not np.isnan(row["atr"]) else 0.0

            # ===== 持仓管理：止损 + 止盈 + Trailing Stop =====
            if position != 0:
                # 移动止损（基于 ATR）
                if atr > 0 and self.trail_atr_mult > 0:
                    if position > 0:
                        new_sl = price - self.trail_atr_mult * atr
                        sl_price = max(sl_price, new_sl)
                    else:
                        new_sl = price + self.trail_atr_mult * atr
                        sl_price = min(sl_price, new_sl)

                exit_flag = False
                if position > 0:
                    if price <= sl_price or price >= tp_price:
                        exit_flag = True
                else:
                    if price >= sl_price or price <= tp_price:
                        exit_flag = True

                if exit_flag:
                    pnl = (price - entry_price) * size * position
                    pnl_total += pnl
                    cash += pnl
                    trades += 1
                    if pnl > 0:
                        wins += 1
                        loss_streak = 0
                    else:
                        loss_streak += 1
                        # 连续亏损到阈值 → 冷静期
                        if loss_streak >= self.max_loss_streak:
                            cooldown_left = self.cooldown_bars
                            loss_streak = 0
                            logger.info(
                                "🧊 %s 连续亏损触发冷静期: %d bars",
                                symbol,
                                cooldown_left,
                            )

                    position = 0
                    size = 0.0
                    entry_price = 0.0
                    sl_price = 0.0
                    tp_price = 0.0

            # ===== 计算账户权益与回撤 =====
            if position != 0:
                equity = cash + (price - entry_price) * size * position
            else:
                equity = cash

            max_equity = max(max_equity, equity)
            if max_equity > 0:
                dd_pct = (equity - max_equity) / max_equity * 100.0
                max_dd_pct = min(max_dd_pct, dd_pct)

            # ===== 空仓状态：是否尝试开仓 =====
            if position == 0:
                # 冷静期中，禁止新开仓
                if cooldown_left > 0:
                    cooldown_left -= 1
                    continue

                # 多因子过滤（趋势 + RSI）
                trend_long_ok = bool(row["trend_long_ok"] and row["rsi_long_ok"])
                trend_short_ok = bool(row["trend_short_ok"] and row["rsi_short_ok"])

                strat_sig = row["strategy_signal"]
                long_signal = trend_long_ok and strat_sig > 0
                short_signal = trend_short_ok and strat_sig < 0

                if not (long_signal or short_signal):
                    continue

                # ATR 必须有效
                if atr <= 0:
                    continue

                # 计算基于 ATR 的 SL/TP 价格
                if long_signal:
                    sl_price_candidate = price - self.sl_atr_mult * atr
                    tp_price_candidate = price + self.tp_atr_mult * atr
                    sl_dist = price - sl_price_candidate
                    tp_dist = tp_price_candidate - price
                else:
                    sl_price_candidate = price + self.sl_atr_mult * atr
                    tp_price_candidate = price - self.tp_atr_mult * atr
                    sl_dist = sl_price_candidate - price
                    tp_dist = price - tp_price_candidate

                if sl_dist <= 0 or tp_dist <= 0:
                    continue

                rr = tp_dist / sl_dist
                if rr < self.min_rr:
                    # 盈亏比不满足要求，不开仓
                    continue

                # 仓位大小 = 每笔风险金额 / 止损距离
                risk_amount = cash * self.risk_per_trade
                if risk_amount <= 0:
                    continue

                size = risk_amount / sl_dist
                if size <= 0:
                    continue

                # 建仓
                position = 1 if long_signal else -1
                entry_price = price
                sl_price = sl_price_candidate
                tp_price = tp_price_candidate

        return SymbolResult(
            pnl=pnl_total, trades=trades, wins=wins, max_dd_pct=max_dd_pct
        )


# ============================================================
# 5. 多币种回测
# ============================================================
def run_backtest(
    symbols: List[str],
    days: int,
    initial_capital: float,
    seed: Optional[int],
    data_source: str,
) -> Dict[str, SymbolResult]:
    logger.info("🚀 SmartBacktest V8 启动")
    logger.info("🪙 币种: %s", symbols)
    logger.info("📅 回测天数: %d", days)
    logger.info("📊 数据源: %s", data_source)

    if seed is not None:
        np.random.seed(seed)

    engine = AdaptiveSignalEngine()
    market = RealMarketData()

    per_capital = initial_capital / len(symbols)

    results: Dict[str, SymbolResult] = {}
    total_pnl = 0.0
    total_trades = 0
    total_wins = 0
    worst_dd_pct = 0.0

    for sym in symbols:
        logger.info("🔍 处理 %s", sym)

        # 获取 K 线数据
        try:
            if data_source == "real":
                df = market.get_recent_klines(sym, "5m", days)
                if df is None or len(df) == 0:
                    logger.warning("⚠️ %s 真实数据为空，使用模拟市场", sym)
                    df = generate_mock_data(sym, days, seed)
            else:
                df = generate_mock_data(sym, days, seed)
        except Exception as e:
            logger.error("❌ 获取 %s 真实数据失败: %s", sym, e)
            df = generate_mock_data(sym, days, seed)

        res = engine.run_symbol_backtest(sym, df, per_capital)

        results[sym] = res
        total_pnl += res.pnl
        total_trades += res.trades
        total_wins += res.wins
        worst_dd_pct = min(worst_dd_pct, res.max_dd_pct)

    win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0.0

    print("\n========== 📈 SmartBacktest V8 报告 ==========")
    print(f"总收益: {total_pnl:.2f}")
    print(f"总交易数: {total_trades}")
    print(f"总胜率: {win_rate:.2f}%")
    print(f"最大回撤: {worst_dd_pct:.2f}%\n")

    print("按币种：")
    for sym, r in results.items():
        print(
            f"- {sym}: pnl={r.pnl:.2f}, trades={r.trades}, "
            f"win_rate={r.win_rate:.2f}%, maxDD={r.max_dd_pct:.2f}%"
        )

    return results


# ============================================================
# 6. main
# ============================================================
def parse_symbols(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="SmartBacktest V8")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC/USDT,ETH/USDT",
        help="逗号分隔的交易对，例如: BTC/USDT,ETH/USDT",
    )
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["real", "mock"],
        default="real",
        help="real=Binance真实数据, mock=模拟K线",
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
