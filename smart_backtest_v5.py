# -*- coding: utf-8 -*-
"""
smart_backtest_v5.py

五哥专用版本（V5）：
- 修复「总收益 ≠ 各币种收益之和」的问题
- 抽佣（保险柜）改为**全局**统计，不再错误归因到单个币种
- 明确拆分：
    ① 策略交易收益（每笔真实盈亏的总和）
    ② 保险柜安全利润（抽佣）
    ③ 账户当前可用资金（继续交易的本金）
- 报告中新增「总资产（含保险柜）」并保证所有数值自洽

运行示例：
    python smart_backtest_v5.py --symbols BTC/USDT,ETH/USDT,SOL/USDT --days 30 --engine ai_prod
    python smart_backtest_v5.py --symbols BTC/USDT,ETH/USDT,SOL/USDT --days 60 --engine baseline --use-real-data

如需兼容之前命令，可直接重命名为 smart_backtest.py 使用。
"""

import argparse
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 尝试导入真实行情模块（如果存在的话）
try:
    from real_market_data import load_for_smart_backtest
except Exception:  # noqa
    load_for_smart_backtest = None


# ===========================
# 日志配置
# ===========================
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("SmartBacktest")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


logger = setup_logger()


# ===========================
# 市场数据模拟 & 真实数据加载
# ===========================
def simulate_market_data(
    symbol: str,
    days: int,
    interval_minutes: int = 5,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """生成简单的随机游走 K 线数据，用于本地快速测试。"""
    if seed is not None:
        np.random.seed(seed)

    bars_per_day = int(24 * 60 / interval_minutes)
    n = days * bars_per_day
    # 时间索引
    idx = pd.date_range(
        end=pd.Timestamp.utcnow(), periods=n, freq=f"{interval_minutes}min"
    )

    # 价格随机游走
    base_price = 20000.0 if "BTC" in symbol.upper() else 1500.0
    returns = np.random.normal(loc=0.0001, scale=0.01, size=n)
    price = base_price * np.exp(np.cumsum(returns))

    # OHLCV
    df = pd.DataFrame(index=idx)
    df["close"] = price
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1) * (
        1 + np.random.uniform(0, 0.002, size=n)
    )
    df["low"] = df[["open", "close"]].min(axis=1) * (
        1 - np.random.uniform(0, 0.002, size=n)
    )
    df["volume"] = np.random.uniform(1, 10, size=n)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "timestamp"}, inplace=True)
    return df


def load_market_data(
    symbol: str,
    days: int,
    use_real: bool,
    interval: str = "5m",
) -> pd.DataFrame:
    """统一的行情获取入口，优先真实数据，失败则退回模拟数据。"""
    if use_real and load_for_smart_backtest is not None:
        try:
            df = load_for_smart_backtest(symbol, days=days, interval=interval)
            if df is None or df.empty:
                logger.warning(f"⚠️ 真实数据为空，回退到模拟数据: {symbol}")
            else:
                logger.info(f"📊 使用真实市场数据: {symbol} ({len(df)} 行)")
                # 确保必要字段存在
                needed = {"timestamp", "open", "high", "low", "close", "volume"}
                missing = needed - set(df.columns)
                if missing:
                    raise ValueError(f"真实数据缺少列: {missing}")
                return df
        except Exception as e:
            logger.error(
                f"❌ 下载真实数据失败({symbol})，原因: {e}"
            )
            logger.warning(f"⚠️ 使用 fallback 模拟数据: {symbol}")

    # 模拟数据
    df_sim = simulate_market_data(symbol, days=days, interval_minutes=5)
    logger.info(f"📊 使用模拟市场数据: {symbol} ({len(df_sim)} 行)")
    return df_sim


# ===========================
# 信号引擎
# ===========================
class BaselineSignalEngine:
    """基础版信号引擎：简单均线 + 趋势过滤。"""

    def __init__(
        self,
        fast_window: int = 20,
        slow_window: int = 60,
        trend_window: int = 48,
        up_threshold: float = 0.002,
        down_threshold: float = -0.002,
    ):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.trend_window = trend_window
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        ma_fast = close.rolling(self.fast_window).mean()
        ma_slow = close.rolling(self.slow_window).mean()

        # 过去 trend_window 根 K 线的涨跌幅（向后看）
        trend = close.pct_change(self.trend_window)

        signal = pd.Series(0, index=df.index, dtype=float)

        long_cond = (ma_fast > ma_slow) & (trend > self.up_threshold)
        short_cond = (ma_fast < ma_slow) & (trend < self.down_threshold)

        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        signal.ffill(inplace=True)
        signal.fillna(0.0, inplace=True)
        return signal


class AISignalEngine:
    """AI 版信号引擎：稍微复杂一点，多因子组合。"""

    def __init__(
        self,
        fast_window: int = 10,
        slow_window: int = 40,
        trend_window: int = 48,
        vol_window: int = 30,
        up_threshold: float = 0.003,
        down_threshold: float = -0.003,
    ):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.trend_window = trend_window
        self.vol_window = vol_window
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]

        ma_fast = close.rolling(self.fast_window).mean()
        ma_slow = close.rolling(self.slow_window).mean()

        # 过去一段时间的趋势
        trend = close.rolling(self.trend_window, min_periods=self.trend_window).apply(
            lambda x: (float(x[-1]) / float(x[0]) - 1.0) if x[0] != 0 else 0.0,
            raw=True,
        )

        # 波动率
        ret = close.pct_change().fillna(0.0)
        vol = ret.rolling(self.vol_window).std()

        # 多因子打分
        score = pd.Series(0.0, index=df.index)
        score += np.tanh((ma_fast - ma_slow) / (ma_slow + 1e-8)) * 0.6
        score += np.tanh(trend / 0.02) * 0.3
        score += np.tanh(-vol / 0.03) * 0.1  # 波动越小越敢做

        signal = pd.Series(0.0, index=df.index)
        signal[score > self.up_threshold] = 1.0
        signal[score < self.down_threshold] = -1.0
        signal.ffill(inplace=True)
        signal.fillna(0.0, inplace=True)
        return signal


# ===========================
# 回测统计结构
# ===========================
@dataclass
class SymbolStats:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    profit: float = 0.0  # 该币种产生的真实盈亏（不含抽佣）
    tech_signals: int = 0
    random_signals: int = 0  # 当前版本不使用随机，但保留字段方便扩展


@dataclass
class BacktestResult:
    initial_equity: float
    final_equity: float
    safe_profit: float
    net_worth: float  # final_equity + safe_profit
    total_strategy_profit: float  # 所有交易真实盈亏之和
    total_trades: int
    total_wins: int
    total_losses: int
    win_rate: float
    max_drawdown: float
    monthly_return: float
    symbol_stats: Dict[str, SymbolStats] = field(default_factory=dict)
    ai_score: float = 0.0
    ai_grade: str = "C"
    ai_comment: str = ""


# ===========================
# 主回测引擎
# ===========================
class SmartBacktest:
    def __init__(
        self,
        symbols: List[str],
        days: int,
        engine_type: str = "baseline",
        use_real_data: bool = False,
        initial_equity: float = 10000.0,
        leverage: float = 3.0,
        risk_per_trade: float = 0.01,  # 每笔风险占当前权益比例
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        max_consec_losses: int = 5,
        profit_step: float = 0.10,  # 每盈利 10% 触发一次抽佣
        skim_pct: float = 0.20,  # 抽取 20% 利润进入保险柜
    ):
        self.symbols = symbols
        self.days = days
        self.use_real_data = use_real_data
        self.initial_equity = initial_equity
        self.equity = initial_equity  # 当前账户资金（可继续交易）
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_consec_losses = max_consec_losses

        # 抽佣 / 保险柜
        self.profit_step = profit_step
        self.skim_pct = skim_pct
        self.safe_profit = 0.0  # 保险柜里的钱（完全隔离）
        self.next_skim_threshold = initial_equity * (1.0 + profit_step)

        # 权益曲线（含保险柜），用于计算最大回撤
        self.equity_curve: List[float] = [initial_equity]

        # 选择信号引擎
        if engine_type == "baseline":
            self.engine = BaselineSignalEngine()
        elif engine_type == "ai_prod":
            self.engine = AISignalEngine()
        else:
            raise ValueError(f"未知引擎类型: {engine_type}")

        # 每个币种统计
        self.symbol_stats: Dict[str, SymbolStats] = {
            sym: SymbolStats() for sym in symbols
        }

        # 全局交易统计
        self.global_trades: List[Tuple[str, float]] = []  # (symbol, pnl)
        self.logger = logger

        self.logger.info("🚀 智能回测系统初始化完成")
        self.logger.info(
            f"💰 初始资金: ${self.initial_equity:,.2f}, 杠杆: {self.leverage:.1f}x, "
            f"使用真实数据: {self.use_real_data}, 引擎: {engine_type}"
        )

    # ---------- 抽佣逻辑（全局） ----------
    def _maybe_skim_profit(self):
        """当总资产（权益+保险柜）超过下一档阈值时，执行抽佣。"""
        net_worth = self.equity + self.safe_profit
        # 当净值超过下一个阈值（以初始资金为步长），每次抽取固定金额
        while net_worth >= self.next_skim_threshold:
            # 本次可抽取利润（以初始资金为基准）
            step_profit = self.initial_equity * self.profit_step
            skim_amount = step_profit * self.skim_pct

            # 防止抽空账户
            skim_amount = min(skim_amount, max(self.equity - self.initial_equity * 0.2, 0))

            if skim_amount <= 0:
                break

            self.equity -= skim_amount
            self.safe_profit += skim_amount

            net_worth = self.equity + self.safe_profit
            self.logger.info(
                f"🏦 触发抽佣: 抽取 ${skim_amount:,.2f} 至保险柜，当前保险柜=${self.safe_profit:,.2f}，"
                f"账户资金=${self.equity:,.2f}"
            )
            # 下一档阈值向上移动一个 step
            self.next_skim_threshold += self.initial_equity * self.profit_step

    # ---------- 单币种回测 ----------
    def _run_single_symbol(self, symbol: str):
        self.logger.info(f"🔍 测试币种: {symbol}")
        df = load_market_data(symbol, days=self.days, use_real=self.use_real_data)

        # 生成信号
        signals = self.engine.generate_signals(df)
        self.symbol_stats[symbol].tech_signals = int((signals != 0).sum())

        position_size = 0.0
        entry_price = 0.0
        notional = 0.0  # 仓位名义价值（用于计算 PnL）
        consec_losses = 0

        for i in range(1, len(df)):
            price = float(df["close"].iloc[i])
            signal = float(signals.iloc[i])

            # 更新权益曲线（按总资产记：可用资金 + 保险柜）
            net_worth = self.equity + self.safe_profit
            self.equity_curve.append(net_worth)

            # 已持仓 -> 判断止盈/止损/反向信号
            if position_size != 0.0:
                pnl_pct = (price - entry_price) / entry_price
                # 多头仓位 PnL
                trade_pnl = notional * pnl_pct

                exit_reason = None
                should_exit = False

                if pnl_pct <= -self.stop_loss_pct:
                    should_exit = True
                    exit_reason = "止损"
                elif pnl_pct >= self.take_profit_pct:
                    should_exit = True
                    exit_reason = "止盈"
                elif signal < 0:
                    should_exit = True
                    exit_reason = "反向信号"

                if should_exit:
                    self.equity += trade_pnl
                    self.global_trades.append((symbol, trade_pnl))
                    st = self.symbol_stats[symbol]
                    st.trades += 1
                    st.profit += trade_pnl

                    if trade_pnl >= 0:
                        st.wins += 1
                        consec_losses = 0
                    else:
                        st.losses += 1
                        consec_losses += 1

                    self.logger.debug(
                        f"🔁 平仓[{symbol}] | 原因={exit_reason} | PnL=${trade_pnl:,.2f} | "
                        f"价格={price:.2f} | 权益=${self.equity:,.2f}"
                    )

                    # 触发抽佣检查
                    self._maybe_skim_profit()

                    # 连续亏损风控
                    if consec_losses >= self.max_consec_losses:
                        self.logger.info(
                            f"🧊 {symbol} 连续亏损 {consec_losses} 笔，停止该币种后续交易。"
                        )
                        position_size = 0.0
                        break

                    # 清空仓位
                    position_size = 0.0
                    entry_price = 0.0
                    notional = 0.0

            # 空仓 -> 根据信号开仓
            if position_size == 0.0 and signal > 0:
                # 以当前权益的 risk_per_trade 开仓，乘以杠杆
                risk_capital = self.equity * self.risk_per_trade
                notional = risk_capital * self.leverage
                if notional <= 0:
                    continue

                position_size = notional / price
                entry_price = price

                self.logger.debug(
                    f"🟢 开多[{symbol}] | 价格={price:.2f} | 名义仓位=${notional:,.2f} | "
                    f"当前权益=${self.equity:,.2f}"
                )

        # 如果最后仍有持仓，按收盘价平仓
        if position_size != 0.0:
            last_price = float(df["close"].iloc[-1])
            pnl_pct = (last_price - entry_price) / entry_price
            trade_pnl = notional * pnl_pct
            self.equity += trade_pnl
            self.global_trades.append((symbol, trade_pnl))

            st = self.symbol_stats[symbol]
            st.trades += 1
            st.profit += trade_pnl
            if trade_pnl >= 0:
                st.wins += 1
            else:
                st.losses += 1

            self._maybe_skim_profit()

    # ---------- AI 打分 ----------
    @staticmethod
    def _compute_ai_score(
        win_rate: float,
        max_drawdown: float,
        monthly_return: float,
    ) -> Tuple[float, str, str]:
        """
        简单版 AI 风险收益评分：
        - 月化收益重要，但不能脱离回撤和胜率
        - 大回撤严重扣分
        """
        score = 0.0

        # 1) 月化收益：20% 月化给到 40 分上限
        if monthly_return > 0:
            score += min(40.0, monthly_return * 200.0)

        # 2) 胜率：>40% 才开始加分，50% 胜率约给 20 分
        if win_rate > 0.4:
            score += min(20.0, (win_rate - 0.4) * 200.0)

        # 3) 最大回撤：无回撤 30 分，上限回撤 60% -> 0 分
        if max_drawdown < 0.6:
            score += (0.6 - max_drawdown) / 0.6 * 30.0

        # 4) 基础分
        score += 10.0

        score = max(0.0, min(100.0, score))

        if score >= 80:
            grade = "A"
            comment = "风险收益匹配良好，可以考虑中等仓位试运行。"
        elif score >= 65:
            grade = "B"
            comment = "表现尚可，但回撤或胜率一般，建议小仓位试运行。"
        elif score >= 50:
            grade = "C"
            comment = "策略风险收益比偏弱，建议先小仓位或仅用作研究参考。"
        else:
            grade = "D"
            comment = "风险较大且收益不稳定，不建议直接用于实盘。"

        return score, grade, comment

    # ---------- 运行主流程 ----------
    def run(self) -> BacktestResult:
        self.logger.info(
            f"🎯 开始智能回测: {self.symbols}, 天数={self.days}"
        )

        for sym in self.symbols:
            self._run_single_symbol(sym)

        # 计算总策略收益（所有交易 PnL 的总和）
        total_strategy_profit = sum(p for _, p in self.global_trades)
        net_worth = self.equity + self.safe_profit

        # 自检：策略收益应等于 总资产(含保险柜) - 初始资金 （数值可能有极小浮动）
        diff_check = (net_worth - self.initial_equity) - total_strategy_profit
        if abs(diff_check) > 1e-6:
            self.logger.warning(
                f"⚠️ 收益自检存在微小偏差: diff={diff_check:.6f}，"
                f"这通常是由于浮点数误差导致。"
            )

        # 计算最大回撤（基于总资产曲线）
        peak = self.equity_curve[0]
        max_dd = 0.0
        for v in self.equity_curve:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        # 统计交易相关指标
        total_trades = len(self.global_trades)
        total_wins = len([1 for _, p in self.global_trades if p > 0])
        total_losses = len([1 for _, p in self.global_trades if p < 0])
        win_rate = total_wins / total_trades if total_trades > 0 else 0.0

        # 月化收益（基于总资产）
        total_return = (net_worth / self.initial_equity - 1.0) if self.initial_equity > 0 else 0.0
        months = self.days / 30.0 if self.days > 0 else 1.0
        monthly_return = total_return / months if months > 0 else 0.0

        # AI 评分
        ai_score, ai_grade, ai_comment = self._compute_ai_score(
            win_rate=win_rate,
            max_drawdown=max_dd,
            monthly_return=monthly_return,
        )

        # === 报告输出 ===
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("🧠 智能量化交易系统 - 回测报告")
        self.logger.info("=" * 80)
        self.logger.info("")
        self.logger.info("📈 智能性能汇总:")
        self.logger.info(f"  测试币种: {len(self.symbols)} 个")
        self.logger.info(f"  总交易次数: {total_trades} 笔")
        self.logger.info(f"  总收益(仅策略交易): ${total_strategy_profit:,.2f}")
        self.logger.info(f"  当前账户资金(可继续交易): ${self.equity:,.2f}")
        self.logger.info(f"  保险柜安全利润(抽佣): ${self.safe_profit:,.2f}")
        self.logger.info(f"  总资产（账户+保险柜）: ${net_worth:,.2f}")
        self.logger.info(f"  平均胜率: {win_rate * 100:.1f}%")
        self.logger.info(f"  最大回撤: {max_dd * 100:.1f}%")
        self.logger.info(
            f"  粗略年化/月化估算: 月化≈{monthly_return * 100:.1f}% （目标≥20%）"
        )
        self.logger.info("")
        self.logger.info("📊 各币种智能表现:")

        for sym in self.symbols:
            st = self.symbol_stats[sym]
            sym_win_rate = st.wins / st.trades * 100 if st.trades > 0 else 0.0
            self.logger.info(
                f"  🟡 {sym}: {st.trades} 笔, 胜率: {sym_win_rate:.1f}%, "
                f"收益: ${st.profit:,.2f}"
            )
            self.logger.info(
                f"     信号来源: 技术={st.tech_signals}, 随机={st.random_signals}"
            )

        self.logger.info("")
        self.logger.info("🏦 利润抽取 + 复利模拟（简化版）:")
        self.logger.info(
            f"  回测期间共抽取到“保险柜”的安全利润≈${self.safe_profit:,.2f}；"
            f"若将这些视作完全风险隔离的收益，剩余账户资金继续用于复利。"
        )
        self.logger.info("")
        self.logger.info("🤖 AI 风险收益评分:")
        self.logger.info(
            f"  综合得分: {ai_score:.1f} / 100, 等级: {ai_grade}, 评语: {ai_comment}"
        )
        self.logger.info("")
        self.logger.info("🎉 智能回测完成！")
        self.logger.info("=" * 80)

        return BacktestResult(
            initial_equity=self.initial_equity,
            final_equity=self.equity,
            safe_profit=self.safe_profit,
            net_worth=net_worth,
            total_strategy_profit=total_strategy_profit,
            total_trades=total_trades,
            total_wins=total_wins,
            total_losses=total_losses,
            win_rate=win_rate,
            max_drawdown=max_dd,
            monthly_return=monthly_return,
            symbol_stats=self.symbol_stats,
            ai_score=ai_score,
            ai_grade=ai_grade,
            ai_comment=ai_comment,
        )


# ===========================
# CLI
# ===========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="智能量化回测 V5")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC/USDT,ETH/USDT,SOL/USDT",
        help="逗号分隔的交易对列表，如: BTC/USDT,ETH/USDT",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="回测天数，如 30 或 60",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="ai_prod",
        choices=["baseline", "ai_prod"],
        help="信号引擎类型: baseline / ai_prod",
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="使用真实行情（需 real_market_data.py 支持）",
    )
    parser.add_argument(
        "--initial-equity",
        type=float,
        default=10000.0,
        help="初始资金",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=3.0,
        help="杠杆倍数",
    )
    parser.add_argument(
        "--risk-per-trade",
        type=float,
        default=0.01,
        help="单笔风险占当前权益比例，例如 0.01 表示 1%%",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=0.02,
        help="单笔止损比例，如 0.02=2%%",
    )
    parser.add_argument(
        "--take-profit-pct",
        type=float,
        default=0.04,
        help="单笔止盈比例，如 0.04=4%%",
    )
    parser.add_argument(
        "--max-consec-losses",
        type=int,
        default=5,
        help="单币种允许的最大连续亏损笔数，超过则暂停该币种交易",
    )
    parser.add_argument(
        "--profit-step",
        type=float,
        default=0.10,
        help="每盈利多少比例（相对初始资金）触发一次抽佣，如 0.10=10%%",
    )
    parser.add_argument(
        "--skim-pct",
        type=float,
        default=0.20,
        help="每次抽佣的比例，例如 0.20 表示抽取 20%% 利润到保险柜",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    bt = SmartBacktest(
        symbols=symbols,
        days=args.days,
        engine_type=args.engine,
        use_real_data=bool(args.use_real_data),
        initial_equity=args.initial_equity,
        leverage=args.leverage,
        risk_per_trade=args.risk_per_trade,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        max_consec_losses=args.max_consec_losses,
        profit_step=args.profit_step,
        skim_pct=args.skim_pct,
    )
    bt.run()


if __name__ == "__main__":
    main()
