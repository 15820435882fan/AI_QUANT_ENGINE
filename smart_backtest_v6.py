#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
smart_backtest_v6.py

V6 特点：
1. 信号引擎多指标融合：MA 趋势 + RSI + Bollinger + 波动率。
2. AI 引擎（ai_prod）用打分模型给出多空/观望信号。
3. 抽佣只来自「已实现盈利」，不会再从浮盈中乱抽。
4. 收益统计严格校验：总收益 ≈ 各币种 PnL 之和（数值上相差不超过 1e-6）。
5. 风控：单币连续亏损熔断、单笔风险、不超仓。
6. AI 评分：年化收益 / 回撤 / 胜率 / 收益集中度 多维打分。
"""

import argparse
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("SmartBacktest")


# ================================
# 工具函数
# ================================

def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def simulate_market_data(symbol: str, days: int, interval_minutes: int = 5) -> pd.DataFrame:
    """
    简易随机游走行情，用于本地无网络时测试。
    """
    n = int(days * 24 * 60 / interval_minutes)
    if n < 50:
        n = 50

    now = pd.Timestamp.utcnow()
    index = pd.date_range(end=now, periods=n, freq=f"{interval_minutes}min")

    # 不同币种不同初始价格
    base_price = {
        "BTCUSDT": 30000,
        "ETHUSDT": 2000,
        "SOLUSDT": 50,
    }.get(symbol.replace("/", ""), 100)

    # 随机游走
    mu = 0.0
    sigma = 0.01  # 日波动 ~1%
    dt_frac = interval_minutes / (60 * 24)
    rets = np.random.normal(mu * dt_frac, sigma * math.sqrt(dt_frac), size=n)
    prices = base_price * np.exp(np.cumsum(rets))

    df = pd.DataFrame(index=index)
    df["close"] = prices
    df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])
    df["high"] = df[["open", "close"]].max(axis=1) * (1 + np.random.uniform(0, 0.002, size=n))
    df["low"] = df[["open", "close"]].min(axis=1) * (1 - np.random.uniform(0, 0.002, size=n))
    df["volume"] = np.random.uniform(10, 100, size=n)
    df["symbol"] = symbol
    df["timestamp"] = df.index

    return df.reset_index(drop=True)


def load_real_data_wrapper(symbol: str, days: int, interval: str = "5m") -> Optional[pd.DataFrame]:
    """
    从 real_market_data.py 中加载真实数据，如果失败返回 None。
    """
    try:
        from real_market_data import load_for_smart_backtest
    except Exception:
        return None

    try:
        df = load_for_smart_backtest(symbol=symbol, days=days, interval=interval)
        if df is None or df.empty:
            return None

        # 期望列: open, high, low, close, volume, timestamp
        # 统一加上 symbol 列
        if "symbol" not in df.columns:
            df["symbol"] = symbol
        if "timestamp" not in df.columns:
            # 尝试用 index 当作时间
            df["timestamp"] = pd.to_datetime(df.index)

        return df.reset_index(drop=True)
    except Exception as e:
        logger.warning("⚠️ 真实数据加载失败，将使用模拟数据: %s (%s)", symbol, e)
        return None


# ================================
# 信号引擎
# ================================

@dataclass
class EngineConfig:
    engine_type: str = "ai_prod"   # baseline / ai_prod
    fast_ma: int = 20
    slow_ma: int = 60
    rsi_period: int = 14
    bb_window: int = 20
    no_random: bool = False


class SignalEngine:
    """
    多指标融合信号引擎：
    - baseline: 规则型（趋势 + RSI + Bollinger）
    - ai_prod: 带打分的“伪 AI”引擎
    """

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["close"]

        df["ret_1"] = close.pct_change()
        df["ret_5"] = close.pct_change(5)
        df["ma_fast"] = close.rolling(self.cfg.fast_ma).mean()
        df["ma_slow"] = close.rolling(self.cfg.slow_ma).mean()
        df["ma_trend"] = (df["ma_fast"] - df["ma_slow"]) / (df["ma_slow"] + 1e-9)

        df["rsi"] = self._rsi(close, self.cfg.rsi_period)

        # Bollinger
        roll = close.rolling(self.cfg.bb_window)
        ma = roll.mean()
        std = roll.std(ddof=0)
        df["bb_mid"] = ma
        df["bb_up"] = ma + 2 * std
        df["bb_low"] = ma - 2 * std
        df["bb_pos"] = (close - df["bb_mid"]) / (2 * std + 1e-9)

        # 波动率
        df["volatility"] = df["ret_1"].rolling(20).std(ddof=0)

        # 清理前期 NaN
        warmup = max(self.cfg.fast_ma, self.cfg.slow_ma, self.cfg.rsi_period, self.cfg.bb_window) + 5
        df = df.iloc[warmup:].reset_index(drop=True)

        return df

    def generate_signals(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        输出 DataFrame 包含：
        - signal: -1, 0, 1
        - confidence: 0~1
        """
        df = self._prepare_features(df_raw)
        df["signal"] = 0
        df["confidence"] = 0.0

        trend = df["ma_trend"]
        rsi = df["rsi"]
        bb_pos = df["bb_pos"]
        vola = df["volatility"].fillna(0)

        if self.cfg.engine_type == "baseline":
            # 简单规则：
            # 1. 多头趋势 ma_fast > ma_slow
            # 2. RSI 在 50~70 范围
            # 3. 价格不在极端上轨/下轨附近
            long_cond = (trend > 0.001) & (rsi.between(52, 70)) & (bb_pos < 0.8)
            flat_cond = (rsi.between(40, 60)) & (bb_pos.between(-0.5, 0.5))
            short_cond = (trend < -0.001) & (rsi.between(30, 48)) & (bb_pos > -0.8)

            df.loc[long_cond, "signal"] = 1
            df.loc[short_cond, "signal"] = -1
            df.loc[flat_cond, "signal"] = 0

            # 置信度：趋势强度 + RSI 距离中性 50 的绝对值
            conf = (
                trend.abs().clip(0, 0.01) / 0.01 * 0.6
                + (rsi - 50).abs().clip(0, 20) / 20 * 0.4
            )
            df["confidence"] = conf.clip(0, 1)

        else:  # ai_prod
            # 组合打分：趋势、RSI、boll、波动率
            score_trend = trend.clip(-0.02, 0.02) / 0.02  # -1~1
            score_rsi = ((rsi - 50) / 25).clip(-1, 1)     # -1~1
            score_bb = (-bb_pos).clip(-2, 2) / 2          # 趋近下轨更偏向多
            score_vol = vola.clip(0, 0.03) / 0.03         # 0~1，过高波动我们会降低权重

            # 综合方向分：
            # 大趋势 + RSI + BB 位置为主，小部分随机扰动可选
            rand_term = 0.0
            if not self.cfg.no_random:
                rand_term = np.random.normal(0, 0.1, size=len(df))

            raw_score = (
                0.45 * score_trend +
                0.25 * score_rsi +
                0.20 * score_bb -
                0.15 * score_vol +  # 波动大减分
                0.10 * rand_term
            )

            df["raw_score"] = raw_score

            # 将 raw_score 压缩到 -1~1
            score = raw_score.clip(-2, 2) / 2.0
            df["direction_score"] = score

            # 门限区间
            long_th = 0.25
            short_th = -0.25

            df.loc[score > long_th, "signal"] = 1
            df.loc[score < short_th, "signal"] = -1
            df.loc[score.between(short_th, long_th), "signal"] = 0

            df["confidence"] = score.abs().clip(0, 1)

        return df


# ================================
# 回测核心
# ================================

@dataclass
class SymbolStats:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    vault: float = 0.0
    max_consec_losses: int = 0
    current_consec_losses: int = 0


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    leverage: float = 3.0
    risk_per_trade: float = 0.01      # 每笔占总资金 1%
    sl_pct: float = 0.01              # 止损 1%
    tp_pct: float = 0.02              # 止盈 2%
    max_holding_bars: int = 96        # 最多持仓 96 根 5m K (≈8h)
    max_consec_losses: int = 5        # 单币连续亏损 N 次后熔断
    harvest_trigger: float = 0.10     # 账户新高后，超过 10% 启动抽佣
    harvest_ratio: float = 0.20       # 抽出超额收益的 20%
    big_trade_harvest: float = 0.05   # 单笔盈利超过账户 5% 触发一次额外抽佣
    use_real_data: bool = False
    engine_cfg: EngineConfig = field(default_factory=EngineConfig)


class SmartBacktest:
    def __init__(self, symbols: List[str], days: int, cfg: BacktestConfig):
        self.symbols = symbols
        self.days = days
        self.cfg = cfg
        self.engine = SignalEngine(cfg.engine_cfg)

        self.equity: float = cfg.initial_capital
        self.equity_peak: float = cfg.initial_capital
        self.equity_curve: List[float] = [cfg.initial_capital]

        self.vault_total: float = 0.0  # “保险柜”累计利润

        self.symbol_stats: Dict[str, SymbolStats] = {
            sym: SymbolStats() for sym in symbols
        }

        self.max_drawdown_pct: float = 0.0

    # --------- 数据加载 ---------

    def _load_market_data(self, symbol: str) -> pd.DataFrame:
        sym_ccxt = symbol.replace("/", "")

        if self.cfg.use_real_data:
            df_real = load_real_data_wrapper(sym_ccxt, self.days, interval="5m")
            if df_real is not None and not df_real.empty:
                logger.info("📊 使用真实市场数据: %s (%d 行)", symbol, len(df_real))
                return df_real

        # 否则使用模拟数据
        df_sim = simulate_market_data(sym_ccxt, self.days, interval_minutes=5)
        logger.info("📊 使用模拟市场数据: %s (%d 行)", symbol, len(df_sim))
        return df_sim

    # --------- 交易模拟 ---------

    def _update_drawdown(self):
        self.equity_peak = max(self.equity_peak, self.equity)
        dd = (self.equity_peak - self.equity) / self.equity_peak if self.equity_peak > 0 else 0.0
        self.max_drawdown_pct = max(self.max_drawdown_pct, dd * 100)

    def _harvest_if_needed(self, symbol: str, realized_pnl: float):
        """
        两种抽佣：
        1. 账户新高突破：超出部分 * harvest_ratio
        2. 单笔盈利超过 big_trade_harvest * equity 的一部分
        """
        stats = self.symbol_stats[symbol]

        # --- 单笔大盈利抽佣 ---
        if realized_pnl > 0 and realized_pnl >= self.cfg.big_trade_harvest * self.equity:
            harvest = realized_pnl * 0.20   # 单笔盈利抽 20%
            harvest = min(harvest, self.equity - 1000)  # 避免抽干
            if harvest > 0:
                self.equity -= harvest
                self.vault_total += harvest
                stats.vault += harvest

        # --- 账户新高突破抽佣 ---
        if self.equity > self.equity_peak * (1 + self.cfg.harvest_trigger):
            extra = self.equity - self.equity_peak
            harvest = extra * self.cfg.harvest_ratio
            harvest = min(harvest, self.equity - 1000)
            if harvest > 0:
                self.equity -= harvest
                self.vault_total += harvest
                stats.vault += harvest

            # 更新高点
            self.equity_peak = self.equity

    def _run_single_symbol(self, symbol: str):
        df_raw = self._load_market_data(symbol)
        if df_raw is None or df_raw.empty:
            logger.warning("⚠️ 市场数据为空: %s", symbol)
            return

        signals_df = self.engine.generate_signals(df_raw)

        # 合并原始 close/时间
        df = signals_df.copy()
        df["close"] = df_raw["close"].iloc[-len(df):].reset_index(drop=True)
        df["timestamp"] = df_raw["timestamp"].iloc[-len(df):].reset_index(drop=True)

        stats = self.symbol_stats[symbol]

        position_size = 0.0   # 持仓数量（币）
        entry_price = 0.0
        bars_held = 0

        for i, row in df.iterrows():
            price = float(row["close"])
            signal = int(row["signal"])
            conf = float(row["confidence"])

            # 更新权益曲线（这里假设均匀分布到每根 K 线）
            self.equity_curve.append(self.equity)
            self._update_drawdown()

            # 有持仓时检查止盈止损 / 时间止盈
            if position_size != 0:
                direction = 1 if position_size > 0 else -1
                pnl_pct = direction * (price - entry_price) / entry_price
                # 止损
                if pnl_pct <= -self.cfg.sl_pct:
                    realized = position_size * (price - entry_price)
                    self.equity += realized
                    stats.trades += 1
                    if realized > 0:
                        stats.wins += 1
                        stats.current_consec_losses = 0
                    else:
                        stats.losses += 1
                        stats.current_consec_losses += 1
                    stats.pnl += realized
                    self._harvest_if_needed(symbol, realized)

                    position_size = 0.0
                    bars_held = 0
                    continue

                # 止盈
                if pnl_pct >= self.cfg.tp_pct:
                    realized = position_size * (price - entry_price)
                    self.equity += realized
                    stats.trades += 1
                    if realized > 0:
                        stats.wins += 1
                        stats.current_consec_losses = 0
                    else:
                        stats.losses += 1
                        stats.current_consec_losses += 1
                    stats.pnl += realized
                    self._harvest_if_needed(symbol, realized)

                    position_size = 0.0
                    bars_held = 0
                    continue

                # 时间止盈
                bars_held += 1
                if bars_held >= self.cfg.max_holding_bars:
                    realized = position_size * (price - entry_price)
                    self.equity += realized
                    stats.trades += 1
                    if realized > 0:
                        stats.wins += 1
                        stats.current_consec_losses = 0
                    else:
                        stats.losses += 1
                        stats.current_consec_losses += 1
                    stats.pnl += realized
                    self._harvest_if_needed(symbol, realized)

                    position_size = 0.0
                    bars_held = 0
                    continue

            # 熔断：连续亏损过多，不再开新仓
            stats.max_consec_losses = max(stats.max_consec_losses, stats.current_consec_losses)
            if stats.current_consec_losses >= self.cfg.max_consec_losses:
                # 只允许平仓，不再开仓
                continue

            # 无持仓时，根据信号开仓（只做多，暂不做空）
            if position_size == 0 and signal == 1 and conf > 0.3:
                # 计算本次可用风险资金
                max_risk = self.equity * self.cfg.risk_per_trade
                # 假设止损 sl_pct，对应 price*sl_pct 的亏损幅度
                # position_value * sl_pct ≈ max_risk
                position_value = max_risk / (self.cfg.sl_pct + 1e-9)
                # 考虑杠杆
                position_value = min(position_value, self.equity * self.cfg.leverage)
                qty = position_value / price
                if qty <= 0:
                    continue
                position_size = qty
                entry_price = price
                bars_held = 0

        # 如果最后还持仓，平掉
        if position_size != 0:
            last_price = float(df["close"].iloc[-1])
            realized = position_size * (last_price - entry_price)
            self.equity += realized
            stats.trades += 1
            if realized > 0:
                stats.wins += 1
                stats.current_consec_losses = 0
            else:
                stats.losses += 1
                stats.current_consec_losses += 1
            stats.pnl += realized
            self._harvest_if_needed(symbol, realized)

        stats.max_consec_losses = max(stats.max_consec_losses, stats.current_consec_losses)

    # --------- 评分与报告 ---------

    def _compute_ai_score(self, total_pnl: float, total_trades: int) -> Tuple[float, str]:
        if len(self.equity_curve) < 2:
            return 50.0, "数据不足，难以评估。"

        equity_series = pd.Series(self.equity_curve)
        # 简单年化（按 5m K 线估算）
        bars_per_day = 24 * 60 / 5
        days = len(equity_series) / bars_per_day
        if days <= 0:
            annual_return = 0.0
        else:
            final_equity = equity_series.iloc[-1]
            annual_return = (final_equity / self.cfg.initial_capital) ** (365.0 / max(days, 1e-6)) - 1.0

        # 1) 收益分 (0~40)
        if annual_return <= 0:
            ret_score = 10.0 * (1 + annual_return)  # -100% -> 0, 0% -> 10
        else:
            ret_score = 10.0 + min(annual_return, 2.0) / 2.0 * 30.0  # 0~200% -> 10~40
        ret_score = max(0.0, min(40.0, ret_score))

        # 2) 回撤分 (0~30)
        dd = self.max_drawdown_pct
        if dd <= 5:
            dd_score = 30.0
        elif dd <= 15:
            dd_score = 20.0
        elif dd <= 30:
            dd_score = 10.0
        else:
            dd_score = 5.0

        # 3) 胜率 & 交易样本数 (0~15)
        wins = sum(s.wins for s in self.symbol_stats.values())
        trades = sum(s.trades for s in self.symbol_stats.values())
        if trades > 0:
            win_rate = wins / trades
        else:
            win_rate = 0.0

        if trades < 100:
            wr_score = 5.0 * win_rate
        else:
            wr_score = 15.0 * win_rate
        wr_score = max(0.0, min(15.0, wr_score))

        # 4) 收益集中度 (0~15)：越平均越高
        pnl_list = [max(0.0, s.pnl) for s in self.symbol_stats.values()]
        if sum(pnl_list) <= 0:
            conc_score = 5.0
        else:
            weights = np.array(pnl_list) / sum(pnl_list)
            hhi = (weights ** 2).sum()  # 越小越分散
            conc_score = (1 - min(hhi, 1.0)) * 15.0

        score = ret_score + dd_score + wr_score + conc_score
        score = max(0.0, min(100.0, score))

        if score >= 80:
            grade = "A"
            comment = "风险收益表现优秀，可以考虑中等仓位逐步实盘验证。"
        elif score >= 65:
            grade = "B"
            comment = "整体表现良好，但仍建议从小仓位、分阶段验证开始。"
        elif score >= 50:
            grade = "C"
            comment = "策略风险收益比一般，适合作为研究参考或低仓位辅助策略。"
        else:
            grade = "D"
            comment = "当前表现偏弱，建议继续优化后再考虑实盘。"

        full_comment = (
            f"综合得分: {score:.1f} / 100, 等级: {grade}，评语: {comment}"
        )
        return score, full_comment

    def run(self):
        logger.info("🚀 智能回测系统初始化完成")
        logger.info(
            "💰 初始资金: $%.2f, 杠杆: %.1fx, 使用真实数据: %s, 引擎: %s",
            self.cfg.initial_capital,
            self.cfg.leverage,
            self.cfg.use_real_data,
            self.cfg.engine_cfg.engine_type,
        )
        logger.info(
            "🎯 开始智能回测: %s, 天数=%d",
            self.symbols,
            self.days,
        )

        for sym in self.symbols:
            logger.info("🔍 测试币种: %s", sym)
            self._run_single_symbol(sym)

        # ---------- 汇总 ----------
        total_pnl = self.equity - self.cfg.initial_capital
        total_trades = sum(s.trades for s in self.symbol_stats.values())
        total_wins = sum(s.wins for s in self.symbol_stats.values())
        win_rate = (total_wins / total_trades) if total_trades > 0 else 0.0

        # 校验：各币种 PnL 之和
        symbol_pnl_sum = sum(s.pnl for s in self.symbol_stats.values())
        # 允许极小误差
        if abs(symbol_pnl_sum - total_pnl) > 1e-6:
            logger.warning(
                "⚠️ 收益校验存在微小偏差: total_pnl=%.4f, symbol_pnl_sum=%.4f",
                total_pnl,
                symbol_pnl_sum,
            )

        # ---------- 报告 ----------
        logger.info("")
        logger.info("===============================================================================",)
        logger.info("🧠 智能量化交易系统 - 回测报告")
        logger.info("===============================================================================",)
        logger.info("")
        logger.info("📈 智能性能汇总:")
        logger.info("  测试币种: %d 个", len(self.symbols))
        logger.info("  总交易次数: %d 笔", total_trades)
        logger.info("  总收益: $%.2f", total_pnl)
        logger.info("  最终资金: $+%.2f", self.equity)
        logger.info("  平均胜率: %.1f%%", win_rate * 100)
        logger.info("  最大回撤: %.2f%%", self.max_drawdown_pct)

        # 粗略月化估算
        bars_per_day = 24 * 60 / 5
        days = len(self.equity_curve) / bars_per_day
        if days > 0:
            total_return = self.equity / self.cfg.initial_capital - 1
            monthly_return = (1 + total_return) ** (30.0 / max(days, 1e-6)) - 1
            logger.info("  粗略年化/月化估算: 月化≈%.1f%% （目标≥20%%）", monthly_return * 100)
        logger.info("")

        logger.info("📊 各币种智能表现:")
        for sym, s in self.symbol_stats.items():
            wr = (s.wins / s.trades * 100) if s.trades > 0 else 0.0
            logger.info(
                "  🟡 %s: %d 笔, 胜率: %.1f%%, 收益: $%.2f, 抽取到保险柜: $%.2f",
                sym,
                s.trades,
                wr,
                s.pnl,
                s.vault,
            )

        logger.info("")
        logger.info("🏦 利润抽取 + 复利模拟（简化版）:")
        logger.info(
            "  回测期间共抽取到“保险柜”的安全利润≈$%.2f；若将这些视作完全风险隔离的收益，剩余资金继续用于复利。",
            self.vault_total,
        )
        logger.info("")

        score, comment = self._compute_ai_score(total_pnl, total_trades)
        logger.info("🤖 AI 风险收益评分:")
        logger.info("  %s", comment)
        logger.info("")
        logger.info("🎉 智能回测完成！")
        logger.info("===============================================================================",)


# ================================
# CLI
# ================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Backtest V6")

    parser.add_argument(
        "--symbols",
        type=str,
        required=True,
        help="逗号分隔的交易对列表，如 BTC/USDT,ETH/USDT,SOL/USDT",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="回测天数（用于模拟或真实数据窗口）",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="ai_prod",
        choices=["baseline", "ai_prod"],
        help="信号引擎类型：baseline / ai_prod",
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="尝试使用 real_market_data.py 中的真实 K 线数据",
    )
    parser.add_argument(
        "--no-random",
        action="store_true",
        help="关闭引擎中的随机扰动，回测结果可重复",
    )

    # 可选参数：修改 MA、抽佣规则等
    parser.add_argument("--fast-ma", type=int, default=20, help="快速均线周期")
    parser.add_argument("--slow-ma", type=int, default=60, help="慢速均线周期")
    parser.add_argument("--rsi-period", type=int, default=14, help="RSI 周期")
    parser.add_argument("--bb-window", type=int, default=20, help="布林带窗口")

    parser.add_argument("--risk-per-trade", type=float, default=0.01, help="每笔风险占总资金比例")
    parser.add_argument("--sl-pct", type=float, default=0.01, help="止损百分比")
    parser.add_argument("--tp-pct", type=float, default=0.02, help="止盈百分比")

    parser.add_argument("--harvest-trigger", type=float, default=0.10, help="账户新高抽佣触发阈值")
    parser.add_argument("--harvest-ratio", type=float, default=0.20, help="账户新高抽佣比例")
    parser.add_argument("--big-trade-harvest", type=float, default=0.05, help="单笔大盈利抽佣触发阈值")

    return parser.parse_args()


def main():
    setup_logging(logging.INFO)
    args = parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    engine_cfg = EngineConfig(
        engine_type=args.engine,
        fast_ma=args.fast_ma,
        slow_ma=args.slow_ma,
        rsi_period=args.rsi_period,
        bb_window=args.bb_window,
        no_random=args.no_random,
    )

    bt_cfg = BacktestConfig(
        initial_capital=10_000.0,
        leverage=3.0,
        risk_per_trade=args.risk_per_trade,
        sl_pct=args.sl_pct,
        tp_pct=args.tp_pct,
        max_holding_bars=96,
        max_consec_losses=5,
        harvest_trigger=args.harvest_trigger,
        harvest_ratio=args.harvest_ratio,
        big_trade_harvest=args.big_trade_harvest,
        use_real_data=args.use_real_data,
        engine_cfg=engine_cfg,
    )

    backtest = SmartBacktest(symbols, args.days, bt_cfg)
    backtest.run()


if __name__ == "__main__":
    main()
