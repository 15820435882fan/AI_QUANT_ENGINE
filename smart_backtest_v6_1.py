#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
smart_backtest_v6_1.py

一个自带「AI 大脑 + 风控 + 抽佣保险柜」的轻量级回测框架（v6.1）。

设计目标：
1. 结构清晰，方便以后接入真实盘口 / 生产级 AI 决策引擎；
2. 交易逻辑相对保守，但不会「几乎不交易」；
3. 抽佣（利润回抽到保险柜）逻辑与账户净值、策略评分逻辑自洽。
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 日志配置
# -----------------------------------------------------------------------------
logger = logging.getLogger("SmartBacktest")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - SmartBacktest - %(levelname)s - %(message)s",
)


# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------
def generate_synthetic_ohlcv(symbol: str, days: int, freq: str = "5min") -> pd.DataFrame:
    """
    生成简单随机游走的模拟 K 线数据。
    - freq 默认 5 分钟，比较接近你之前实盘/回测的频率；
    - 随机种子固定（按 symbol），便于复现。
    """
    minutes_per_day = int(24 * 60 / 5)  # 5min 频率
    n = days * minutes_per_day
    if n < 200:
        n = 200

    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    # 随机游走 + 轻微趋势
    drift = rng.normal(loc=0.00002, scale=0.00001)  # 日内微弱趋势
    vol = 0.002  # 单根波动

    rets = rng.normal(loc=drift, scale=vol, size=n)
    price0 = 100.0
    prices = price0 * np.exp(np.cumsum(rets))

    # 构造 OHLCV
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq=freq)
    df = pd.DataFrame(index=idx)
    df["close"] = prices
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1) * (1 + rng.normal(0.0005, 0.0005, size=n))
    df["low"] = df[["open", "close"]].min(axis=1) * (1 - rng.normal(0.0005, 0.0005, size=n))
    df["volume"] = rng.lognormal(mean=3, sigma=0.5, size=n)

    return df


def compute_max_drawdown(equity: pd.Series) -> float:
    """
    计算最大回撤（返回 0~1 的正数）。
    """
    if len(equity) < 2:
        return 0.0
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max.replace(0, np.nan)
    max_dd = dd.min()
    if pd.isna(max_dd):
        return 0.0
    return float(-max_dd)


def approx_monthly_return(total_return: float, days: int) -> float:
    """
    用复利方式把 total_return（整段期间）折算成月化。
    """
    if days <= 0:
        return 0.0
    # 把整段看作 N 天，折算成「30 天的等效收益」
    return (1.0 + total_return) ** (30.0 / days) - 1.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# -----------------------------------------------------------------------------
# 简单指标 & AI 风格信号引擎
# -----------------------------------------------------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50.0)


@dataclass
class Signal:
    action: int  # 1 = buy, -1 = sell/close, 0 = hold
    confidence: float  # 0 ~ 1
    reason: str


class BaseEngine:
    def generate_signal(self, df: pd.DataFrame, t: int, has_position: bool) -> Signal:
        raise NotImplementedError


@dataclass
class BaselineEngine(BaseEngine):
    fast: int = 10
    slow: int = 40
    threshold: float = 0.002  # 0.2% 上下穿越才算有效信号
    rsi_low: float = 35.0
    rsi_high: float = 65.0

    def generate_signal(self, df: pd.DataFrame, t: int, has_position: bool) -> Signal:
        if t < max(self.fast, self.slow) + 5:
            return Signal(0, 0.0, "warmup")

        window = df.iloc[: t + 1]
        close = window["close"]
        ema_fast = ema(close, self.fast)
        ema_slow = ema(close, self.slow)
        rsi_val = rsi(close).iloc[-1]

        fast_now = float(ema_fast.iloc[-1])
        slow_now = float(ema_slow.iloc[-1])
        fast_prev = float(ema_fast.iloc[-2])
        slow_prev = float(ema_slow.iloc[-2])

        rel_diff_now = (fast_now - slow_now) / slow_now
        rel_diff_prev = (fast_prev - slow_prev) / slow_prev

        # 多头入场：均线金叉 + rsi 没有极端超买
        if not has_position:
            if rel_diff_prev <= -self.threshold and rel_diff_now >= self.threshold and rsi_val > self.rsi_low:
                conf = clamp(abs(rel_diff_now) / (self.threshold * 3), 0.2, 1.0)
                return Signal(1, conf, "ema_cross_up")

        # 多头离场：死叉或者 rsi 超买
        if has_position:
            if rel_diff_prev >= self.threshold and rel_diff_now <= -self.threshold:
                conf = clamp(abs(rel_diff_now) / (self.threshold * 3), 0.2, 1.0)
                return Signal(-1, conf, "ema_cross_down")
            if rsi_val >= self.rsi_high:
                conf = clamp((rsi_val - self.rsi_high) / 20.0, 0.2, 1.0)
                return Signal(-1, conf, "rsi_overbought")

        return Signal(0, 0.0, "hold")


@dataclass
class AIProdEngine(BaseEngine):
    """
    模拟「AI 生产大脑」：
    - 以 BaselineEngine 为基础；
    - 结合趋势强度 / 波动率 / RSI 形态，给出 0~1 的「AI 信心」。
    """
    base: BaselineEngine = field(default_factory=BaselineEngine)

    def generate_signal(self, df: pd.DataFrame, t: int, has_position: bool) -> Signal:
        if t < max(self.base.fast, self.base.slow) + 20:
            return Signal(0, 0.0, "warmup")

        window = df.iloc[max(0, t - 200): t + 1]
        close = window["close"]
        ret_lookback = 50

        if len(window) < ret_lookback + 5:
            return Signal(0, 0.0, "warmup")

        # 1）调用 baseline 获取原始方向
        base_sig = self.base.generate_signal(df, t, has_position)

        # 2）趋势强度：过去 ret_lookback 根的收益
        trend_return = float(close.iloc[-1] / close.iloc[-ret_lookback] - 1.0)

        # 3）波动率：过去 ret_lookback 根收益的标准差
        returns = close.pct_change().dropna()
        vol = float(returns.tail(ret_lookback).std() or 0.0)

        # 4）RSI 形态：是否处于「温和区间」
        rsi_val = float(rsi(close).iloc[-1])

        # 组合成「AI 风格信心」：趋势好 + 波动合适 + rsi 合理
        score_trend = clamp((trend_return * 5.0) + 0.5, 0.0, 1.0)  # 趋势 20%+ 视为高分
        score_vol = 1.0 - clamp((vol - 0.01) / 0.03, 0.0, 1.0)      # 波动过大或过小都会扣分
        score_rsi = 1.0 - abs(rsi_val - 55.0) / 55.0                # 55 左右最舒服

        ai_conf = clamp(0.4 * score_trend + 0.3 * score_vol + 0.3 * score_rsi, 0.0, 1.0)

        # 没有方向就直接返回 hold，但带上信心供上层参考
        if base_sig.action == 0:
            return Signal(0, ai_conf * 0.5, "ai_hold")

        # 有方向时，用 AI 信心调节强度
        base_sig.confidence = clamp((base_sig.confidence + ai_conf) / 2.0, 0.0, 1.0)
        base_sig.reason = f"{base_sig.reason}|ai"
        return base_sig


# -----------------------------------------------------------------------------
# 交易 & 回测核心
# -----------------------------------------------------------------------------
@dataclass
class Trade:
    symbol: str
    side: str  # "long"
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_pct: float


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    leverage: float = 3.0

    position_fraction: float = 0.25      # 单笔最多占用账户净值的 25%
    max_exposure_fraction: float = 0.8   # 总持仓不超过 80% * leverage

    sl_pct: float = 0.03                 # 单笔止损 3%
    tp_pct: float = 0.09                 # 初始止盈 9%

    max_daily_drawdown: float = 0.05     # 单日最大回撤 5%
    max_total_drawdown: float = 0.35     # 允许的整体最大回撤 35%
    max_consec_losses_symbol: int = 5    # 单币连续亏损 N 笔后冷静期
    symbol_cooldown_trades: int = 10     # 冷静期长度：跳过 N 笔信号

    vault_trigger: float = 0.10          # 净值相对上次高点收益 >10% 时触发抽佣
    vault_fraction: float = 0.20         # 抽取 20% 利润进保险柜

    min_trades_for_score: int = 30       # 少于该交易笔数，评分会打折


class SmartBacktest:
    def __init__(self,
                 symbols: List[str],
                 days: int,
                 engine_name: str = "ai_prod",
                 use_real_data: bool = False,
                 config: Optional[BacktestConfig] = None) -> None:
        self.symbols = symbols
        self.days = days
        self.use_real_data = use_real_data
        self.config = config or BacktestConfig()

        if engine_name == "baseline":
            self.engine: BaseEngine = BaselineEngine()
        elif engine_name == "ai_prod":
            self.engine = AIProdEngine()
        else:
            raise ValueError(f"未知引擎类型: {engine_name}")

        # 账户状态
        self.initial_capital = self.config.initial_capital
        self.cash = self.initial_capital
        self.vault = 0.0  # 保险柜里的安全利润
        self.equity_history: List[Tuple[pd.Timestamp, float]] = []

        self.positions: Dict[str, Dict] = {}  # symbol -> {size, entry_price, entry_time}
        self.trades: List[Trade] = []

        self.symbol_stats: Dict[str, Dict[str, float]] = {}
        self.symbol_consec_losses: Dict[str, int] = {s: 0 for s in symbols}
        self.symbol_cooldown_left: Dict[str, int] = {s: 0 for s in symbols}

        self.global_max_equity = self.initial_capital
        self.global_min_equity = self.initial_capital

        logger.info("🚀 智能回测系统初始化完成")
        logger.info(
            "💰 初始资金: $%.2f, 杠杆: %.1fx, 使用真实数据: %s, 引擎: %s",
            self.initial_capital,
            self.config.leverage,
            self.use_real_data,
            engine_name,
        )

    # ------------------------------------------------------------------
    # 市场数据加载（当前环境无外网，只保留模拟数据实现）
    # ------------------------------------------------------------------
    def load_ohlcv(self, symbol: str) -> pd.DataFrame:
        # 这里保留钩子：未来可以接 real_market_data.load_for_smart_backtest
        # 当前环境你那边连 Binance 有时会被墙，这里默认使用模拟数据。
        df = generate_synthetic_ohlcv(symbol, self.days, freq="5min")
        logger.info("📊 使用模拟市场数据: %s (%d 行)", symbol, len(df))
        return df

    # ------------------------------------------------------------------
    # 账户 & 风控相关辅助
    # ------------------------------------------------------------------
    def _update_equity(self, timestamp: pd.Timestamp, price_map: Dict[str, float]) -> float:
        equity = self.cash
        for sym, pos in self.positions.items():
            px = price_map.get(sym)
            if px is None:
                continue
            equity += pos["size"] * px  # long only

        self.global_max_equity = max(self.global_max_equity, equity)
        self.global_min_equity = min(self.global_min_equity, equity)
        self.equity_history.append((timestamp, equity))
        return equity

    def _current_exposure(self, price_map: Dict[str, float]) -> float:
        exposure = 0.0
        for sym, pos in self.positions.items():
            px = price_map.get(sym)
            if px is None:
                continue
            exposure += abs(pos["size"] * px)
        return exposure

    def _apply_vault_logic(self, equity: float) -> None:
        """
        当（ equity + vault ）超过历史高点一定比例时，抽取部分利润进 vault，
        同时从账户现金中扣除同样金额，模拟「利润回抽到安全账户」的行为。
        """
        total = equity + self.vault
        if total <= self.global_max_equity * (1.0 + self.config.vault_trigger):
            return

        # 超过高点的「利润」
        profit_over_peak = total - self.global_max_equity
        to_vault = profit_over_peak * self.config.vault_fraction

        # 不能把账户掏空
        to_vault = min(to_vault, self.cash * 0.5)
        if to_vault <= 0:
            return

        self.cash -= to_vault
        self.vault += to_vault
        self.global_max_equity = total  # 更新总高点
        logger.info("💰 触发抽佣: 抽取 %.2f 美金进保险柜，当前保险柜余额: %.2f", to_vault, self.vault)

    def _check_global_drawdown_stop(self, equity: float) -> bool:
        total_dd = 1.0 - equity / self.global_max_equity if self.global_max_equity > 0 else 0.0
        if total_dd >= self.config.max_total_drawdown:
            logger.warning("🛑 触发全局最大回撤限制，停止后续所有交易。")
            return True
        return False

    # ------------------------------------------------------------------
    # 核心回测逻辑
    # ------------------------------------------------------------------
    def _run_symbol(self, symbol: str) -> None:
        df = self.load_ohlcv(symbol)
        if df.empty:
            logger.warning("⚠️ %s 数据为空，跳过。", symbol)
            return

        # 记录该 symbol 当天的 first/last index，用于日内回撤控制
        df = df.copy()
        df["date"] = df.index.date

        has_position = False
        local_stop_all = False
        today = None
        day_equity_start = None
        day_max_equity = None

        for t in range(len(df)):
            row = df.iloc[t]
            ts = df.index[t]
            price = float(row["close"])
            price_map = {symbol: price}

            # 每根 K 更新账户净值
            equity = self._update_equity(ts, price_map)
            if self._check_global_drawdown_stop(equity):
                return

            # 日内回撤控制
            cur_day = row["date"]
            if today != cur_day:
                today = cur_day
                day_equity_start = equity
                day_max_equity = equity
            else:
                day_max_equity = max(day_max_equity, equity)
                if day_max_equity > 0:
                    day_dd = 1.0 - equity / day_max_equity
                    if day_dd >= self.config.max_daily_drawdown:
                        logger.warning("🧊 %s 当日回撤达到 %.1f%%，暂停当日剩余交易。", symbol, day_dd * 100)
                        local_stop_all = True

            # 触发抽佣逻辑（基于总净值）
            self._apply_vault_logic(equity)

            # 冷静期/风控：跳过信号
            if local_stop_all:
                continue
            if self.symbol_cooldown_left[symbol] > 0:
                self.symbol_cooldown_left[symbol] -= 1
                continue

            # 处理持仓的止损/止盈
            pos = self.positions.get(symbol)
            if pos is not None:
                entry_price = pos["entry_price"]
                ret = price / entry_price - 1.0
                if ret <= -self.config.sl_pct:
                    self._close_position(symbol, ts, price, reason="stop_loss")
                    has_position = False
                    continue
                if ret >= self.config.tp_pct:
                    self._close_position(symbol, ts, price, reason="take_profit")
                    has_position = False
                    continue
                has_position = True
            else:
                has_position = False

            # AI / Baseline 产生信号
            sig = self.engine.generate_signal(df, t, has_position)
            if sig.action == 0 or sig.confidence <= 0.2:
                continue

            # 开仓 or 平仓
            if sig.action == 1 and not has_position:
                self._open_position(symbol, ts, price, sig)
                has_position = True
            elif sig.action == -1 and has_position:
                self._close_position(symbol, ts, price, reason=sig.reason)
                has_position = False

        # 收尾：强制平掉剩余持仓
        pos = self.positions.get(symbol)
        if pos is not None:
            ts = df.index[-1]
            price = float(df["close"].iloc[-1])
            self._close_position(symbol, ts, price, reason="end_of_test")

    def _open_position(self, symbol: str, ts: pd.Timestamp, price: float, sig: Signal) -> None:
        # 已有仓位就不再加仓（目前一币只允许一笔）
        if symbol in self.positions:
            return

        # 风险：总曝光限制
        price_map = {symbol: price}
        current_exposure = self._current_exposure(price_map)
        max_exposure = self.initial_capital * self.config.leverage * self.config.max_exposure_fraction
        if current_exposure >= max_exposure:
            return

        # 计算本次下单规模：账户净值 * position_fraction * 信心
        # 简化：用当前现金近似净值
        notional = self.cash * self.config.position_fraction * sig.confidence
        notional = min(notional, max_exposure - current_exposure)
        if notional <= 0:
            return

        size = notional / price
        self.cash -= notional  # 全额从现金里扣出去
        self.positions[symbol] = {
            "size": size,
            "entry_price": price,
            "entry_time": ts,
        }

    def _close_position(self, symbol: str, ts: pd.Timestamp, price: float, reason: str) -> None:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return

        size = pos["size"]
        entry_price = pos["entry_price"]
        entry_time = pos["entry_time"]

        notional_entry = size * entry_price
        notional_exit = size * price
        pnl = notional_exit - notional_entry
        ret_pct = pnl / notional_entry if notional_entry != 0 else 0.0

        self.cash += notional_exit

        # 记录交易
        self.trades.append(
            Trade(
                symbol=symbol,
                side="long",
                entry_time=entry_time,
                exit_time=ts,
                entry_price=entry_price,
                exit_price=price,
                size=size,
                pnl=pnl,
                return_pct=ret_pct,
            )
        )

        # 更新连续亏损计数 & 冷静期
        if pnl < 0:
            self.symbol_consec_losses[symbol] += 1
        else:
            self.symbol_consec_losses[symbol] = 0

        if self.symbol_consec_losses[symbol] >= self.config.max_consec_losses_symbol:
            self.symbol_cooldown_left[symbol] = self.config.symbol_cooldown_trades
            logger.warning(
                "🧊 %s 连续亏损 %d 笔，进入冷静期 (%d 笔信号)。",
                symbol,
                self.symbol_consec_losses[symbol],
                self.config.symbol_cooldown_trades,
            )
            self.symbol_consec_losses[symbol] = 0

    # ------------------------------------------------------------------
    # 回测执行 & 报告
    # ------------------------------------------------------------------
    def run(self) -> None:
        logger.info(
            "🎯 开始智能回测: %s, 天数=%d",
            self.symbols,
            self.days,
        )
        for sym in self.symbols:
            logger.info("🔍 测试币种: %s", sym)
            self._run_symbol(sym)

        if not self.equity_history:
            logger.warning("⚠️ 没有产生任何净值记录，可能完全没有成交。")
            return

        equity_series = pd.Series(
            [e for _, e in self.equity_history],
            index=[t for t, _ in self.equity_history],
        )
        max_dd = compute_max_drawdown(equity_series)
        final_equity = equity_series.iloc[-1]
        total_return = final_equity / self.initial_capital - 1.0

        # 统计
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t.pnl > 0)
        winrate = wins / total_trades if total_trades > 0 else 0.0

        # 分币种统计
        symbol_pnl = {s: 0.0 for s in self.symbols}
        symbol_trades = {s: 0 for s in self.symbols}
        for tr in self.trades:
            symbol_pnl[tr.symbol] += tr.pnl
            symbol_trades[tr.symbol] += 1

        days = max(1, self.days)
        mret = approx_monthly_return(total_return, days)

        # AI 风险收益评分
        score = self._compute_ai_score(
            total_return=total_return,
            max_dd=max_dd,
            winrate=winrate,
            trade_count=total_trades,
            months=days / 30.0,
        )
        grade = self._grade_from_score(score)

        # -------------------- 报告输出 --------------------
        logger.info("")
        logger.info("=" * 79)
        logger.info("🧠 智能量化交易系统 - 回测报告")
        logger.info("=" * 79)
        logger.info("")
        logger.info("📈 智能性能汇总:")
        logger.info("  测试币种: %d 个", len(self.symbols))
        logger.info("  总交易次数: %d 笔", total_trades)
        logger.info("  总收益: $%.2f", total_return * self.initial_capital)
        logger.info("  最终资金: $+%.2f", final_equity)
        logger.info("  平均胜率: %.1f%%", winrate * 100)
        logger.info("  最大回撤: %.1f%%", max_dd * 100)
        logger.info("  粗略年化/月化估算: 月化≈%.1f%% （目标≥20%%）", mret * 100)
        logger.info("")
        logger.info("📊 各币种智能表现:")
        for sym in self.symbols:
            trades_sym = symbol_trades.get(sym, 0)
            if trades_sym == 0:
                logger.info("  🟡 %s: 无成交", sym)
                continue
            pnl_sym = symbol_pnl[sym]
            wins_sym = sum(1 for t in self.trades if t.symbol == sym and t.pnl > 0)
            winrate_sym = wins_sym / trades_sym if trades_sym > 0 else 0.0
            logger.info(
                "  🟡 %s: %d 笔, 胜率: %.1f%%, 收益: $%.2f",
                sym,
                trades_sym,
                winrate_sym * 100,
                pnl_sym,
            )
        logger.info("")
        logger.info("🏦 利润抽取 + 复利模拟（简化版）:")
        logger.info(
            "  回测结束时账户资金≈$%.2f，保险柜安全利润≈$%.2f，合计总资产≈$%.2f。",
            final_equity,
            self.vault,
            final_equity + self.vault,
        )
        logger.info("")
        logger.info("🤖 AI 风险收益评分:")
        logger.info("  综合得分: %.1f / 100, 等级: %s, 评语: %s", score, grade, self._comment_from_grade(grade))
        logger.info("")
        logger.info("🎉 智能回测完成！")
        logger.info("=" * 79)

    # ------------------------------------------------------------------
    # 策略打分
    # ------------------------------------------------------------------
    def _compute_ai_score(
        self,
        total_return: float,
        max_dd: float,
        winrate: float,
        trade_count: int,
        months: float,
    ) -> float:
        """
        一个偏「风险控制」导向的综合评分：
        - 收益高但回撤特别大，不会拿到高分；
        - 交易太少或胜率极低，同样被打折。
        """
        # 期望：月化 20% 左右，对应 total_return_target 约：
        total_return_target = (1.0 + 0.20) ** months - 1.0
        total_return_target = max(total_return_target, 0.10)  # 至少 10%

        score_ret = clamp(total_return / total_return_target, 0.0, 2.0) * 100.0
        # 回撤：<=15% 视为优秀，>50% 逐渐归零
        if max_dd <= 0:
            score_dd = 100.0
        elif max_dd <= 0.15:
            score_dd = 100.0
        elif max_dd >= 0.5:
            score_dd = 10.0
        else:
            score_dd = 100.0 * (1.0 - (max_dd - 0.15) / (0.5 - 0.15))

        # 胜率：考虑到盈亏比通常 >1，只要胜率 >45% 就不错
        if trade_count == 0:
            score_win = 0.0
        else:
            if winrate <= 0.35:
                score_win = 20.0 * (winrate / 0.35)
            elif winrate >= 0.65:
                score_win = 100.0
            else:
                score_win = 40.0 + 60.0 * (winrate - 0.35) / (0.65 - 0.35)

        # 交易次数：太少说明尚未「验证」，太多可能是过度交易
        if trade_count < self.config.min_trades_for_score:
            factor = trade_count / self.config.min_trades_for_score
            score_trades = 40.0 * factor
        elif trade_count > 2000:
            score_trades = 60.0 * (2000.0 / trade_count)
        else:
            score_trades = 80.0

        # 组合权重：收益 40%，回撤 30%，胜率 20%，交易次数 10%
        score = (
            0.4 * score_ret
            + 0.3 * score_dd
            + 0.2 * score_win
            + 0.1 * score_trades
        )
        return clamp(score, 0.0, 100.0)

    def _grade_from_score(self, score: float) -> str:
        if score >= 85:
            return "A+"
        if score >= 75:
            return "A"
        if score >= 65:
            return "B"
        if score >= 50:
            return "C"
        if score >= 35:
            return "D"
        return "E"

    def _comment_from_grade(self, grade: str) -> str:
        if grade in ("A+", "A"):
            return "收益与风险平衡较好，可以考虑小资金实盘验证。"
        if grade == "B":
            return "表现不错，但仍有回撤或稳定性方面的提升空间。"
        if grade == "C":
            return "策略风险收益比一般，建议继续优化或仅做研究参考。"
        if grade == "D":
            return "风险偏高或稳定性不足，仅适合作为反向/辅助指标。"
        return "当前策略不建议用于真实资金，可用于反向情绪或继续调参。"


# -----------------------------------------------------------------------------
# CLI 入口
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart AI Backtest v6.1")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC/USDT,ETH/USDT,SOL/USDT",
        help="逗号分隔的交易对，例如: BTC/USDT,ETH/USDT",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="回测天数（用于模拟数据）",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="ai_prod",
        choices=["baseline", "ai_prod"],
        help="信号引擎: baseline 或 ai_prod",
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="预留参数：未来接入真实 K 线。目前环境下仍使用模拟数据。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    bt = SmartBacktest(
        symbols=symbols,
        days=args.days,
        engine_name=args.engine,
        use_real_data=args.use_real_data,
    )
    bt.run()


if __name__ == "__main__":
    main()
