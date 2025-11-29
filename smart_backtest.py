import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Tuple

import numpy as np
import pandas as pd


# 日志配置
logger = logging.getLogger("SmartBacktest")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - SmartBacktest - %(levelname)s - %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)


# ======================================================================
# 信号结构体
# ======================================================================
@dataclass
class SmartSignal:
    signal: str  # BUY / SELL / HOLD
    source: str  # technical / ai / random / none
    strength: float  # 0~1 置信度


# ======================================================================
# Baseline：简单均线突破信号引擎
# ======================================================================
class SmartSignalDetector:
    """智能混合信号检测器：技术信号 + 随机探索信号（Baseline 引擎）"""

    def __init__(
        self,
        use_technical: bool = True,
        use_random: bool = True,
        random_state: int = 42,
        fast_window: int = 10,
        slow_window: int = 30,
        up_threshold: float = 0.01,
        down_threshold: float = 0.01,
    ):
        # 配置项
        self.use_technical = use_technical
        self.use_random = use_random
        self.rng = np.random.default_rng(random_state)

        # 技术参数
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.up_threshold = up_threshold      # 比如 0.01 = 向上突破 1%
        self.down_threshold = down_threshold  # 比如 0.01 = 向下跌破 1%

        # 统计信息：每个 symbol 的技术 / 随机信号触发次数
        self.tech_counts: Dict[str, int] = {}
        self.rand_counts: Dict[str, int] = {}

    def _ensure_counters(self, symbol: str):
        if symbol not in self.tech_counts:
            self.tech_counts[symbol] = 0
        if symbol not in self.rand_counts:
            self.rand_counts[symbol] = 0

    def get_signal(self, symbol: str, history: pd.DataFrame, idx: int) -> SmartSignal:
        """
        根据历史数据生成当前 bar 的交易信号（Baseline 版本）
        """
        self._ensure_counters(symbol)

        if idx < max(self.fast_window, self.slow_window):
            return SmartSignal(signal="HOLD", source="none", strength=0.0)

        window = history.iloc[: idx + 1]
        close = window["close"]

        # ---------- 技术信号 ----------
        if self.use_technical:
            fast_ma = close.rolling(window=self.fast_window).mean().iloc[-1]
            slow_ma = close.rolling(window=self.slow_window).mean().iloc[-1]
            price = close.iloc[-1]

            # 简单突破 + 均线趋势逻辑
            ma_trend = fast_ma - slow_ma
            strong_up = (price > fast_ma * (1 + self.up_threshold)) and (ma_trend > 0)
            strong_down = (price < fast_ma * (1 - self.down_threshold)) and (ma_trend < 0)

            if strong_up:
                self.tech_counts[symbol] += 1
                strength = float(min(abs(ma_trend / price) * 200, 1.0))
                return SmartSignal(signal="BUY", source="technical", strength=strength)

            if strong_down:
                self.tech_counts[symbol] += 1
                strength = float(min(abs(ma_trend / price) * 200, 1.0))
                return SmartSignal(signal="SELL", source="technical", strength=strength)

        # ---------- 随机探索信号 ----------
        if self.use_random:
            # 每隔 30 根 K，且当前没有明显技术信号时，做一次随机探索
            if idx % 30 == 0:
                self.rand_counts[symbol] += 1
                side = self.rng.choice(["BUY", "SELL"])
                strength = float(self.rng.uniform(0.3, 0.8))
                return SmartSignal(signal=side, source="random", strength=strength)

        return SmartSignal(signal="HOLD", source="none", strength=0.0)


# ======================================================================
# AI 风格多指标信号引擎（自包含“大脑”）
# ======================================================================
class AISignalEngine:
    """
    AI 风格信号引擎：组合 MA + MACD + RSI + 波动过滤
    不依赖外部策略文件，作为一个“生产 AI 大脑”的轻量版适配器。
    """

    def __init__(
        self,
        fast_ma: int = 10,
        slow_ma: int = 30,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        min_confidence: float = 0.4,
    ):
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.min_confidence = min_confidence

        # 统计信号数量（对齐 baseline）
        self.tech_counts: Dict[str, int] = {}
        self.rand_counts: Dict[str, int] = {}  # 这里不做随机探索，保持接口一致

    def _ensure_counters(self, symbol: str):
        if symbol not in self.tech_counts:
            self.tech_counts[symbol] = 0
        if symbol not in self.rand_counts:
            self.rand_counts[symbol] = 0

    def _calc_rsi(self, close: pd.Series, period: int) -> float:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean().iloc[-1]
        avg_loss = loss.rolling(period).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    def get_signal(self, symbol: str, history: pd.DataFrame, idx: int) -> SmartSignal:
        """
        多指标组合打分：
        - MA 趋势
        - MACD 柱子方向
        - RSI 超买超卖
        - 波动过滤（标准差过小不交易）
        """
        self._ensure_counters(symbol)

        min_window = max(
            self.slow_ma, self.macd_slow, self.macd_signal + self.macd_slow, self.rsi_period
        )
        if idx < min_window:
            return SmartSignal(signal="HOLD", source="none", strength=0.0)

        window = history.iloc[: idx + 1]
        close = window["close"]

        # ---- MA 趋势 ----
        fast_ma = close.rolling(self.fast_ma).mean().iloc[-1]
        slow_ma = close.rolling(self.slow_ma).mean().iloc[-1]
        price = close.iloc[-1]
        ma_trend = fast_ma - slow_ma

        ma_score = 0.0
        if ma_trend > 0 and price > fast_ma:
            ma_score = +1.0
        elif ma_trend < 0 and price < fast_ma:
            ma_score = -1.0

        # ---- MACD ----
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=self.macd_signal, adjust=False).mean()
        hist = macd - signal_line
        macd_hist = hist.iloc[-1]

        macd_score = 0.0
        if macd_hist > 0:
            macd_score = +1.0
        elif macd_hist < 0:
            macd_score = -1.0

        # ---- RSI ----
        rsi_val = self._calc_rsi(close, self.rsi_period)
        rsi_score = 0.0
        if rsi_val < self.rsi_oversold:
            rsi_score = +1.0
        elif rsi_val > self.rsi_overbought:
            rsi_score = -1.0

        # ---- 波动过滤 ----
        vol = close.pct_change().rolling(20).std().iloc[-1]
        if vol is not None and vol < 0.002:  # 波动太小，不值得出手
            return SmartSignal(signal="HOLD", source="ai", strength=0.0)

        # ---- 综合打分 ----
        total_score = 0.5 * ma_score + 0.3 * macd_score + 0.2 * rsi_score

        if total_score > 0.4:
            self.tech_counts[symbol] += 1
            strength = float(min(total_score, 1.0))
            if strength < self.min_confidence:
                return SmartSignal(signal="HOLD", source="ai", strength=strength)
            return SmartSignal(signal="BUY", source="ai", strength=strength)

        if total_score < -0.4:
            self.tech_counts[symbol] += 1
            strength = float(min(-total_score, 1.0))
            if strength < self.min_confidence:
                return SmartSignal(signal="HOLD", source="ai", strength=strength)
            return SmartSignal(signal="SELL", source="ai", strength=strength)

        return SmartSignal(signal="HOLD", source="ai", strength=0.0)


# ======================================================================
# 主体：SmartBacktest
# ======================================================================
class SmartBacktest:
    """智能回测系统 - 支持真实数据 / 模拟数据 + Baseline / AI 两种大脑"""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: int = 3,
        compound_mode: bool = True,
        use_real_data: bool = False,
        data_loader: Optional[Callable[[str, int], pd.DataFrame]] = None,
        engine_type: str = "baseline",  # baseline / ai_prod
    ):
        """
        :param use_real_data: 是否使用真实历史数据（True 时优先使用 data_loader）
        :param data_loader:   可调用对象：data_loader(symbol, days) -> DataFrame
                              DataFrame 至少包含 ['timestamp','open','high','low','close','volume']
        :param engine_type:   "baseline" 使用 SmartSignalDetector，
                              "ai_prod"  使用 AISignalEngine
        """
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.compound_mode = compound_mode

        self.use_real_data = use_real_data
        self.data_loader = data_loader
        self.engine_type = engine_type

        # Baseline 引擎：默认技术信号开启，随机信号关闭
        self.signal_detector = SmartSignalDetector(
            use_technical=True,
            use_random=False,
            fast_window=10,
            slow_window=30,
            up_threshold=0.01,
            down_threshold=0.01,
        )

        # AI 信号引擎（多指标组合）
        self.ai_engine = AISignalEngine()

        logger.info("🚀 智能回测系统初始化完成")
        logger.info(
            "💰 初始资金: $%s, 杠杆: %sx, 使用真实数据: %s, 引擎: %s",
            f"{initial_capital:,.2f}",
            leverage,
            use_real_data,
            engine_type,
        )

    # ------------------------------------------------------------------ #
    # 数据获取：真实数据优先，不行再生成模拟数据
    # ------------------------------------------------------------------ #
    def _get_data(self, symbol: str, days: int) -> pd.DataFrame:
        # 1）尝试真实数据
        if self.use_real_data and self.data_loader is not None:
            try:
                df = self.data_loader(symbol, days)
                if df is not None and not df.empty:
                    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
                    missing = [c for c in required_cols if c not in df.columns]
                    if not missing:
                        if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
                            df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df = df.sort_values("timestamp").reset_index(drop=True)
                        logger.info("✅ 使用真实历史数据: %s (%d 条)", symbol, len(df))
                        return df
                    else:
                        logger.warning("⚠️ 真实数据缺少列 %s，回退到模拟数据: %s", missing, symbol)
                else:
                    logger.warning("⚠️ 真实数据为空，回退到模拟数据: %s", symbol)
            except Exception as e:
                logger.error("❌ 加载真实数据失败 (%s): %s，回退到模拟数据", symbol, e)

        # 2）使用智能模拟数据
        return self._generate_smart_data(symbol, days)

    def _generate_smart_data(self, symbol: str, days: int) -> pd.DataFrame:
        """生成带趋势 + 波动的模拟 K 线数据（简化版本）"""
        minutes = days * 24  # 粗略：每小时一根
        ts = [datetime.now() - timedelta(hours=minutes - i) for i in range(minutes)]

        base_price = 100.0
        prices = [base_price]
        rng = np.random.default_rng(123)

        for _ in range(1, minutes):
            drift = rng.normal(0, 0.05)
            shock = rng.normal(0, 1.0)
            price = max(1.0, prices[-1] * (1 + drift / 100) + shock)
            prices.append(price)

        prices = np.array(prices)
        high = prices * (1 + rng.uniform(0.0, 0.01, size=len(prices)))
        low = prices * (1 - rng.uniform(0.0, 0.01, size=len(prices)))
        open_ = prices + rng.normal(0, 0.3, size=len(prices))
        close = prices + rng.normal(0, 0.3, size=len(prices))
        volume = rng.integers(100, 1000, size=len(prices))

        df = pd.DataFrame(
            {
                "timestamp": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
        logger.info("✅ 生成 %s 模拟数据: %d 条", symbol, len(df))
        return df

    # ------------------------------------------------------------------ #
    # 回测主流程
    # ------------------------------------------------------------------ #
    def run_smart_backtest(self, symbols: List[str], days: int = 30) -> None:
        logger.info("🎯 开始智能回测: %s, 天数=%d", symbols, days)

        all_results: Dict[str, Dict[str, float]] = {}
        total_trades = 0
        total_pnl = 0.0

        symbol_signal_stats: Dict[str, Tuple[int, int]] = {}

        n_symbols = len(symbols) if symbols else 1
        capital_per_symbol = self.initial_capital / n_symbols

        for symbol in symbols:
            logger.info("🔍 测试币种: %s", symbol)
            data = self._get_data(symbol, days)
            result = self._backtest_single_symbol(symbol, data, starting_capital=capital_per_symbol)

            all_results[symbol] = result
            total_trades += int(result["trades"])
            total_pnl += float(result["pnl"])

            if self.engine_type == "ai_prod":
                tech_count = self.ai_engine.tech_counts.get(symbol, 0)
                rand_count = self.ai_engine.rand_counts.get(symbol, 0)
            else:
                tech_count = self.signal_detector.tech_counts.get(symbol, 0)
                rand_count = self.signal_detector.rand_counts.get(symbol, 0)
            symbol_signal_stats[symbol] = (tech_count, rand_count)

        final_capital = self.initial_capital + total_pnl

        # 计算平均胜率
        win_rates = [res["win_rate"] for res in all_results.values()]
        avg_win_rate = float(np.mean(win_rates)) if win_rates else 0.0

        # 按 days 粗略折算月化收益（以 30 天为一个月）
        gross_return = (final_capital / self.initial_capital) - 1.0
        if days > 0:
            monthly_return_est = gross_return * (30.0 / days)
        else:
            monthly_return_est = gross_return

        # 模拟“每盈利 10% 抽取 20% 利润 + 80% 复利”的分段效果（基于终点近似）
        skim_info = self._simulate_profit_skimming(final_capital, threshold=0.10, skim_ratio=0.20)

        logger.info("")
        logger.info("=" * 80)
        logger.info("🧠 智能量化交易系统 - 回测报告")
        logger.info("=" * 80)
        logger.info("")
        logger.info("📈 智能性能汇总:")
        logger.info("  测试币种: %d个", len(symbols))
        logger.info("  总交易次数: %d笔", total_trades)
        logger.info("  总收益: $%+.2f", total_pnl)
        logger.info("  最终资金: $%+.2f", final_capital)
        logger.info("  平均胜率: %.1f%%", avg_win_rate * 100.0)
        logger.info("  粗略年化/月化估算: 月化≈%.1f%% （目标≥20%%）", monthly_return_est * 100.0)
        logger.info("")
        logger.info("📊 各币种智能表现:")

        for symbol in symbols:
            res = all_results[symbol]
            trades = int(res["trades"])
            win_rate = res["win_rate"] * 100.0
            pnl = res["pnl"]
            tech_cnt, rand_cnt = symbol_signal_stats[symbol]

            logger.info(
                "  🟡 %s: %d笔, 胜率: %.1f%%, 收益: $%+.2f", symbol, trades, win_rate, pnl
            )
            logger.info("     信号来源: 技术=%d, 随机=%d", tech_cnt, rand_cnt)

        logger.info("")
        logger.info("🏦 利润抽取 + 复利模拟（终点近似计算）:")
        logger.info(
            "  若按照“每盈利10%%抽取20%%利润”的规则，本次回测理论可触发 %d 次抽取，",
            skim_info["skim_times"],
        )
        logger.info(
            "  累计可安全落袋利润≈$%+.2f，调整后复利本金≈$%+.2f",
            skim_info["total_withdrawn"],
            skim_info["final_base"],
        )

        logger.info("")
        logger.info("💡 智能优化建议:")
        if monthly_return_est >= 0.20:
            logger.info("  ✅ 粗略月化收益已达到 20%%+ 目标，可以重点评估回撤与稳定性。")
        elif total_pnl > 0:
            logger.info(
                "  ⚖️ 策略盈利但月化尚未达到 20%%，建议优化入场/止盈规则或适度提高仓位。"
            )
        else:
            logger.info("  ⚠️ 当前策略整体亏损，建议调整信号阈值、止损规则，并缩小仓位继续观察。")

        logger.info("")
        logger.info("🎉 智能回测完成！")
        logger.info("=" * 80)

    # ------------------------------------------------------------------ #
    # 利润抽取 + 复利 近似模拟
    # ------------------------------------------------------------------ #
    def _simulate_profit_skimming(
        self,
        final_capital: float,
        threshold: float = 0.10,
        skim_ratio: float = 0.20,
    ) -> Dict[str, float]:
        """
        基于终点资金，对“每盈利threshold抽取 skim_ratio 利润 + 80% 继续复利”的效果做一个近似计算。

        假设资金单调上涨，仅用于评估策略达标后的资金管理效果上限。
        """
        base = self.initial_capital
        total_withdrawn = 0.0
        skim_times = 0

        while final_capital >= base * (1.0 + threshold):
            profit_block = base * threshold            # 本阶段利润 = 10% * base
            withdraw = profit_block * skim_ratio       # 抽取 20% 利润
            compound = profit_block * (1.0 - skim_ratio)  # 剩余 80% 计入本金

            total_withdrawn += withdraw
            base = base + compound
            skim_times += 1

        return {
            "skim_times": skim_times,
            "total_withdrawn": total_withdrawn,
            "final_base": base,
        }

    # ------------------------------------------------------------------ #
    # 单币种回测（带资金 & 风控管理）
    # ------------------------------------------------------------------ #
    def _backtest_single_symbol(
        self,
        symbol: str,
        data: pd.DataFrame,
        starting_capital: float,
    ) -> Dict[str, float]:
        """
        对单个 symbol 进行回测，返回统计结果。

        引入资金管理：
        - 每笔使用固定比例风险（risk_per_trade_pct）
        - 本地资金出现 ≥8% 回撤或连续N次亏损，则停止该币种交易（冷静期）
        - 当本地资金相对初始资金盈利 ≥8% 时，也停止该币种交易（当日止盈）
        """
        position = 0.0  # 持仓数量（正为多，负为空）
        entry_price = 0.0

        equity = starting_capital
        peak_equity = starting_capital

        pnl = 0.0
        trades = 0
        wins = 0
        consecutive_losses = 0

        risk_per_trade_pct = 0.015    # 每笔 1.5% 风险预算
        max_drawdown_stop_pct = 0.08  # 本地资金回撤 8% 停止
        max_profit_stop_pct = 0.08    # 本地资金盈利 8% 停止
        max_consec_losses = 5         # 连续 5 笔亏损停止

        for idx in range(len(data)):
            row = data.iloc[idx]
            price = float(row["close"])

            # 选择使用哪个大脑
            if self.engine_type == "ai_prod":
                signal = self.ai_engine.get_signal(symbol, data, idx)
            else:
                signal = self.signal_detector.get_signal(symbol, data, idx)

            # 平仓逻辑
            if position != 0:
                if position > 0:
                    # 多单止损 / 止盈
                    if price <= entry_price * 0.97 or price >= entry_price * 1.05:
                        trade_pnl = (price - entry_price) * position
                        equity += trade_pnl
                        pnl += trade_pnl
                        trades += 1
                        if trade_pnl > 0:
                            wins += 1
                            consecutive_losses = 0
                        else:
                            consecutive_losses += 1
                        position = 0
                else:
                    # 空单止损 / 止盈
                    if price >= entry_price * 1.03 or price <= entry_price * 0.95:
                        trade_pnl = (entry_price - price) * abs(position)
                        equity += trade_pnl
                        pnl += trade_pnl
                        trades += 1
                        if trade_pnl > 0:
                            wins += 1
                            consecutive_losses = 0
                        else:
                            consecutive_losses += 1
                        position = 0

                # 更新峰值 & 风控检查
                if equity > peak_equity:
                    peak_equity = equity
                drawdown = peak_equity - equity

                if drawdown >= starting_capital * max_drawdown_stop_pct:
                    logger.info(
                        "  🧊 %s 触发回撤止损（%.2f%%），停止该币种交易。",
                        symbol,
                        max_drawdown_stop_pct * 100.0,
                    )
                    break

                if equity - starting_capital >= starting_capital * max_profit_stop_pct:
                    logger.info(
                        "  🎯 %s 触发当日止盈（%.2f%%），停止该币种交易。",
                        symbol,
                        max_profit_stop_pct * 100.0,
                    )
                    break

                if consecutive_losses >= max_consec_losses:
                    logger.info(
                        "  🧊 %s 连续亏损 %d 笔，停止该币种交易。",
                        symbol,
                        max_consec_losses,
                    )
                    break

            # 开仓逻辑
            if position == 0 and signal.signal in ("BUY", "SELL") and signal.strength > 0:
                trade_capital = equity * risk_per_trade_pct * self.leverage
                if trade_capital <= 0:
                    continue

                qty = trade_capital / price
                if signal.signal == "BUY":
                    position = qty
                    entry_price = price
                elif signal.signal == "SELL":
                    position = -qty
                    entry_price = price

        # 平掉最后的持仓（按最后价格）
        if position != 0 and len(data) > 0:
            last_price = float(data["close"].iloc[-1])
            if position > 0:
                trade_pnl = (last_price - entry_price) * position
            else:
                trade_pnl = (entry_price - last_price) * abs(position)
            equity += trade_pnl
            pnl += trade_pnl
            trades += 1
            if trade_pnl > 0:
                wins += 1

        win_rate = (wins / trades) if trades > 0 else 0.0
        return {
            "pnl": pnl,
            "trades": trades,
            "win_rate": win_rate,
        }


# ======================================================================
# CLI 入口
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="智能高频交易回测系统（SmartBacktest）")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT,SOL/USDT")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--leverage", type=int, default=3)
    parser.add_argument("--no-random", action="store_true", help="关闭随机信号（baseline 引擎用）")
    parser.add_argument("--no-technical", action="store_true", help="关闭技术信号（不推荐）")
    parser.add_argument("--use-real-data", action="store_true", help="使用真实历史K线数据")
    parser.add_argument(
        "--engine",
        type=str,
        default="baseline",
        choices=["baseline", "ai_prod"],
        help="选择信号引擎：baseline=简单均线大脑，ai_prod=多指标AI大脑",
    )
    parser.add_argument("--fast-ma", type=int, default=10, help="快速均线窗口长度")
    parser.add_argument("--slow-ma", type=int, default=30, help="慢速均线窗口长度")
    parser.add_argument("--up-threshold", type=float, default=0.01, help="向上突破阈值(如0.01=1%)")
    parser.add_argument("--down-threshold", type=float, default=0.01, help="向下跌破阈值(如0.01=1%)")

    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    # 真实数据 loader：尝试调用 real_market_data.RealMarketData，失败就回退到模拟数据
    def real_data_loader(symbol: str, days: int) -> pd.DataFrame:
        """
        统一从 real_market_data.load_for_smart_backtest 取数据，
        由 real_market_data.py 保证返回标准格式。
        """
        try:
            from real_market_data import load_for_smart_backtest
        except Exception as e:
            logger.error("❌ 无法从 real_market_data 导入 load_for_smart_backtest: %s", e)
            return pd.DataFrame()

        try:
            df = load_for_smart_backtest(symbol, days)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            logger.error("❌ 调用 load_for_smart_backtest 失败 (%s): %s", symbol, e)
            return pd.DataFrame()

            return df if df is not None else pd.DataFrame()
        except Exception as e:
            logger.error("❌ real_market_data 加载失败 (%s): %s，回退到模拟数据", symbol, e)
            return pd.DataFrame()

    if args.use_real_data:
        data_loader = real_data_loader
    else:
        # 返回空 DataFrame，触发回退到模拟数据
        def dummy_loader(symbol: str, days: int) -> pd.DataFrame:
            return pd.DataFrame()

        data_loader = dummy_loader

    backtest = SmartBacktest(
        initial_capital=args.capital,
        leverage=args.leverage,
        use_real_data=args.use_real_data,
        data_loader=data_loader,
        engine_type=args.engine,
    )

    # 配置信号检测器（仅对 baseline 引擎生效）
    backtest.signal_detector.use_technical = not args.no_technical
    backtest.signal_detector.use_random = not args.no_random

    backtest.signal_detector.fast_window = args.fast_ma
    backtest.signal_detector.slow_window = args.slow_ma
    backtest.signal_detector.up_threshold = args.up_threshold
    backtest.signal_detector.down_threshold = args.down_threshold

    # 运行回测
    backtest.run_smart_backtest(symbols, days=args.days)


if __name__ == "__main__":
    main()
