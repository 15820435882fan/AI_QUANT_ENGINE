import argparse
import logging
import math
import random
from typing import Dict, List

import numpy as np
import pandas as pd


# =============================
# 指标计算
# =============================

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """简单 RSI 计算，缺失值用 50 填补，避免前期 NaN 干扰逻辑。"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR 用于波动和止损/止盈距离。"""
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
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


# =============================
# 模拟市场数据
# =============================

def generate_synthetic_ohlc(days: int, freq_per_day: int = 288, seed: int = None,
                            start_price: float = 100.0) -> pd.DataFrame:
    """
    生成带有一定趋势 + 波动聚集特征的 5m 级别模拟 K 线。
    仅用于离线调试交易框架，不用于真实策略评估。
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n = days * freq_per_day
    sigma = 0.004  # 基础波动率

    prices = [start_price]
    for _ in range(1, n):
        # 简单的波动 regime：高低波动交替
        vol = sigma * (0.5 + 1.5 * random.random())
        ret = np.random.normal(0, vol)
        prices.append(prices[-1] * (1 + ret))

    prices = np.array(prices)
    close = prices
    open_ = np.concatenate([[prices[0]], prices[:-1]])
    high = np.maximum(open_, close) * (1 + np.random.uniform(0, 0.0015, size=n))
    low = np.minimum(open_, close) * (1 - np.random.uniform(0, 0.0015, size=n))

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )


# =============================
# AI 风格信号引擎（Route A：集中提升信号质量）
# =============================

class AIEngineV2:
    """
    一个“多因子合成”的 AI 风格信号引擎：
    - 使用 fast / slow MA 判断趋势
    - 使用 RSI 控制节奏（避免极端超买/超卖）
    - 要求价格回踩/反弹到 fast MA 附近才参与，减少追涨杀跌
    """

    def __init__(self, fast: int = 20, slow: int = 60, rsi_period: int = 14):
        self.fast = fast
        self.slow = slow
        self.rsi_period = rsi_period

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        close = df["close"]
        fast = close.rolling(self.fast).mean()
        slow = close.rolling(self.slow).mean()
        rsi = compute_rsi(close, self.rsi_period)
        atr = compute_atr(df, 14)

        signals = np.zeros(len(df), dtype=int)

        for i in range(self.slow, len(df)):
            if (
                math.isnan(fast.iat[i])
                or math.isnan(slow.iat[i])
                or math.isnan(rsi.iat[i])
                or math.isnan(atr.iat[i])
            ):
                continue

            price = close.iat[i]
            f = fast.iat[i]
            s = slow.iat[i]
            r = rsi.iat[i]
            a = atr.iat[i]
            if a <= 0:
                continue

            bullish = f > s * 1.0005
            bearish = f < s * 0.9995

            # 价格相对短均线的偏离程度（回踩/反弹过滤器）
            dist = (price - f) / price

            # 逻辑：
            #   - 有趋势（fast vs slow）
            #   - 价格回到 fast MA 附近（|dist| < 0.3%）
            #   - RSI 不在极端区间（40~60），避免过度追高或接飞刀
            long_setup = bullish and (abs(dist) < 0.003) and (40 <= r <= 60)
            short_setup = bearish and (abs(dist) < 0.003) and (40 <= r <= 60)

            if long_setup and not short_setup:
                signals[i] = 1
            elif short_setup and not long_setup:
                signals[i] = -1
            else:
                signals[i] = 0

        return signals


class BaselineEngine:
    """对比用：简单均线交叉引擎。"""

    def __init__(self, fast: int = 20, slow: int = 60):
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        close = df["close"]
        fast = close.rolling(self.fast).mean()
        slow = close.rolling(self.slow).mean()
        signals = np.zeros(len(df), dtype=int)

        for i in range(self.slow, len(df)):
            if math.isnan(fast.iat[i]) or math.isnan(slow.iat[i]):
                continue
            if fast.iat[i] > slow.iat[i] * 1.0005:
                signals[i] = 1
            elif fast.iat[i] < slow.iat[i] * 0.9995:
                signals[i] = -1
            else:
                signals[i] = 0
        return signals


# =============================
# 单币种回测引擎
# =============================

def backtest_symbol(
    df: pd.DataFrame,
    signals: np.ndarray,
    initial_equity: float,
    risk_pct: float = 0.01,
    fee_rate: float = 0.0005,
    atr_mult_sl: float = 1.5,
    atr_mult_tp: float = 2.5,
    max_consec_losses: int = 5,
) -> Dict[str, float]:
    """
    对单个 symbol 进行回测：
    - 每笔交易风险固定为账户权益的 risk_pct
    - 止损/止盈基于 ATR（R:R 大约 1:1.5 ~ 1:2）
    - 累计最大回撤 & 交易统计
    - 简化版利润“保险柜”机制（仅本 symbol 统计，方便汇总）
    """
    equity = initial_equity
    locker = 0.0
    position = 0  # 0: 无仓, 1: 多, -1: 空
    entry_price = 0.0
    qty = 0.0
    sl = 0.0
    tp = 0.0
    consec_losses = 0
    trades = 0
    wins = 0
    pnl_sum = 0.0

    equity_history: List[float] = []
    peak_equity = equity
    max_dd = 0.0

    close = df["close"].values
    low = df["low"].values
    high = df["high"].values
    atr = compute_atr(df, 14).values

    for i in range(len(df)):
        price = close[i]

        # 记录权益 & 实时更新最大回撤
        equity_history.append(equity)
        if equity > peak_equity:
            peak_equity = equity
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if drawdown > max_dd:
            max_dd = drawdown

        # 有持仓时，优先处理风控平仓
        if position != 0:
            exit_reason = None
            exit_price = price

            if position == 1:  # 多头
                if low[i] <= sl:
                    exit_price = sl
                    exit_reason = "SL"
                elif high[i] >= tp:
                    exit_price = tp
                    exit_reason = "TP"
            else:  # 空头
                if high[i] >= sl:
                    exit_price = sl
                    exit_reason = "SL"
                elif low[i] <= tp:
                    exit_price = tp
                    exit_reason = "TP"

            # 信号反向，做一次“翻仓式”平/反
            if exit_reason is None and signals[i] == -position:
                exit_price = price
                exit_reason = "flip"

            if exit_reason is not None:
                trades += 1
                gross = (exit_price - entry_price) * qty * (1 if position == 1 else -1)
                fees = fee_rate * (abs(entry_price * qty) + abs(exit_price * qty))
                pnl = gross - fees

                equity += pnl
                pnl_sum += pnl

                if pnl > 0:
                    wins += 1
                    consec_losses = 0
                else:
                    consec_losses += 1

                position = 0
                qty = 0.0

                # 连亏熔断：该币种后面不再交易
                if consec_losses >= max_consec_losses:
                    break

                # 简单利润抽取机制（仅针对本 symbol，方便统计）
                if equity > peak_equity * 1.10:
                    profit_over = equity - peak_equity
                    skim = profit_over * 0.2
                    equity -= skim
                    locker += skim
                    peak_equity = equity

                continue  # 本根 K 线已经平仓，下一根再看开仓

        # 无持仓：尝试根据信号开仓
        if position == 0 and signals[i] != 0 and not math.isnan(atr[i]) and atr[i] > 0:
            side = signals[i]
            a = atr[i]
            if side == 1:
                sl_price = price - atr_mult_sl * a
                tp_price = price + atr_mult_tp * a
            else:
                sl_price = price + atr_mult_sl * a
                tp_price = price - atr_mult_tp * a

            if tp_price <= 0 or sl_price <= 0:
                continue

            risk_per_unit = abs(price - sl_price)
            if risk_per_unit <= 0:
                continue

            capital_at_risk = equity * risk_pct
            q = capital_at_risk / risk_per_unit
            if q <= 0:
                continue

            position = side
            entry_price = price
            qty = q
            sl = sl_price
            tp = tp_price

    return {
        "equity": float(equity),
        "pnl": float(pnl_sum),
        "trades": int(trades),
        "wins": int(wins),
        "win_rate": float(wins / trades * 100) if trades > 0 else 0.0,
        "max_dd": float(max_dd * 100),  # 转成百分比
        "locker": float(locker),
    }


# =============================
# AI 风险收益评分
# =============================

def ai_score(total_return_pct: float, max_dd_pct: float, trades: int) -> (float, str, str):
    """
    根据收益 / 回撤 / 交易样本数量给出一个 0~100 的简单评分。
    这里只是“策略体检”，不是严格的量化评价。
    """
    score = 50.0  # 基础分

    # 收益因子：20% 收益大约 +5 分，100% 收益大约 +20 分，封顶 +25
    if total_return_pct > 0:
        score += min(25.0, total_return_pct / 4.0)
    else:
        score += max(-25.0, total_return_pct / 4.0)  # 亏损扣分

    # 回撤因子：回撤越大扣分越多，40% 回撤约 -20 分，极端情况最多 -30
    score -= min(30.0, max_dd_pct * 0.5)

    # 样本量因子：太少的交易样本不可信
    if trades < 30:
        score -= 10.0
    elif trades < 100:
        score -= 5.0

    # 边界 & 评级
    score = max(0.0, min(100.0, score))

    if score >= 80:
        grade = "A"
        comment = "收益与风控表现优秀，可以考虑小资金逐步放大验证。"
    elif score >= 65:
        grade = "B"
        comment = "收益和风险较均衡，可在严格风控下小规模试用。"
    elif score >= 50:
        grade = "C"
        comment = "策略风险收益比一般，建议先小仓位或仅用作研究参考。"
    elif score >= 35:
        grade = "D"
        comment = "策略稳定性较差，不建议直接用于真实资金。"
    else:
        grade = "E"
        comment = "当前策略不建议用于真实资金，可用于反向情绪或继续调参。"

    return score, grade, comment


# =============================
# 主回测流程
# =============================

def run_backtest(
    symbols: List[str],
    days: int,
    engine_name: str,
    initial_balance: float = 10000.0,
):
    logger = logging.getLogger("SmartBacktest")

    if engine_name == "ai_prod":
        engine = AIEngineV2()
    elif engine_name == "baseline":
        engine = BaselineEngine()
    else:
        raise ValueError(f"未知引擎类型: {engine_name}")

    n = len(symbols)
    per_symbol_equity = initial_balance / n if n > 0 else initial_balance

    logger.info("🚀 开始回测 ...")
    all_results: Dict[str, Dict[str, float]] = {}

    for idx, sym in enumerate(symbols):
        logger.info("🔍 测试币种: %s", sym)
        # 为不同 symbol 使用不同 seed，避免完全相同的价格轨迹
        seed = 100 + idx * 17
        df = generate_synthetic_ohlc(days=days, seed=seed)
        logger.info("📊 使用模拟市场数据: %s", sym)

        signals = engine.generate_signals(df)
        res = backtest_symbol(df, signals, initial_equity=per_symbol_equity)
        all_results[sym] = res

    # 汇总统计
    total_trades = sum(r["trades"] for r in all_results.values())
    total_pnl = sum(r["pnl"] for r in all_results.values())
    total_locker = sum(r["locker"] for r in all_results.values())
    final_equity = initial_balance + total_pnl
    avg_win_rate = (
        np.mean([r["win_rate"] for r in all_results.values() if r["trades"] > 0])
        if total_trades > 0
        else 0.0
    )
    max_dd_pct = max(r["max_dd"] for r in all_results.values()) if all_results else 0.0

    total_return_pct = (final_equity + total_locker - initial_balance) / initial_balance * 100.0

    score, grade, comment = ai_score(total_return_pct, max_dd_pct, total_trades)

    # 输出报告
    logger.info("=" * 80)
    logger.info("🧠 智能量化交易系统 - 回测报告")
    logger.info("=" * 80)
    logger.info("测试币种: %d 个", len(symbols))
    logger.info("总交易次数: %d 笔", total_trades)
    logger.info("总收益: $%.2f", total_pnl)
    logger.info("最终资金: $%.2f (账户: %.2f + 保险柜: %.2f)", final_equity, final_equity, total_locker)
    logger.info("平均胜率: %.1f%%", avg_win_rate)
    logger.info("最大回撤: %.1f%%", max_dd_pct)
    logger.info("总收益率(含保险柜): %.1f%%", total_return_pct)
    logger.info("")
    logger.info("📊 各币种表现:")
    for sym, r in all_results.items():
        logger.info(
            "  🟡 %s: %d 笔, 胜率: %.1f%%, 收益: $%.2f, 抽取到保险柜: $%.2f, 最大回撤: %.1f%%",
            sym,
            r["trades"],
            r["win_rate"],
            r["pnl"],
            r["locker"],
            r["max_dd"],
        )
    logger.info("")
    logger.info("🏦 利润抽取 + 复利模拟（简化版）:")
    logger.info(
        "  回测结束时账户资金≈$%.2f，保险柜安全利润≈$%.2f，合计总资产≈$%.2f。",
        final_equity,
        total_locker,
        final_equity + total_locker,
    )
    logger.info("")
    logger.info("🤖 AI 风险收益评分:")
    logger.info("  综合得分: %.1f / 100, 等级: %s, 评语: %s", score, grade, comment)
    logger.info("")
    logger.info("🎉 智能回测完成！")
    logger.info("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="Smart AI Backtest v6.4")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC/USDT,ETH/USDT,SOL/USDT",
        help="逗号分隔的交易对列表，如 BTC/USDT,ETH/USDT",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="回测天数（使用 5m 模拟 K 线，每天约 288 根）",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="ai_prod",
        choices=["ai_prod", "baseline"],
        help="信号引擎类型：ai_prod 或 baseline",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=10000.0,
        help="初始总资金，默认 10000 美元",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - SmartBacktest - %(levelname)s - %(message)s",
    )
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    run_backtest(
        symbols=symbols,
        days=args.days,
        engine_name=args.engine,
        initial_balance=args.initial_balance,
    )


if __name__ == "__main__":
    main()
