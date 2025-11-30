#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
smart_backtest_v4.py

一个自洽的「智能回测 + 资金管理」小总成：
- 支持模拟数据或真实 K 线（如果 real_market_data 提供的话）
- 两种引擎：baseline / ai_prod（目前逻辑相同，只是为将来接入生产 AI 大脑预留接口）
- 单向做多 + 固定止损止盈 + 多级风控
- 利润抽取到“保险柜”，同时保留一部分用于复利
- 给出一个简单的 AI 风险收益评分（0-100）

注意：
- 这是一个简化版回测内核，目的是让逻辑清晰、稳定可跑，
  以后再在这个基础上迭代复杂度（多策略、多周期、多品种协同等）。
"""

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

# ===================== 日志初始化 =====================

LOG_FORMAT = "%(asctime)s - SmartBacktest - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SmartBacktest")


# ===================== 工具函数 =====================

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """确保 DataFrame 以 datetime 索引。"""
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        else:
            # 没有时间列就假定是等间隔，自己造一个
            df = df.copy()
            df.index = pd.date_range(
                start=pd.Timestamp("2020-01-01"),
                periods=len(df),
                freq="5min",
            )
    return df


def generate_synthetic_ohlcv(symbol: str, days: int, freq: str = "5min") -> pd.DataFrame:
    """生成一个平稳随机游走价序列，用来做烟雾测试。"""
    bars = days * 24 * 60 // 5  # 5 分钟 K
    rng = np.random.default_rng(seed=hash(symbol) % (2**32 - 1))

    # 做一个缓慢随机游走 + 一点趋势
    steps = rng.normal(loc=0.0002, scale=0.01, size=bars)
    price = 1 + np.cumsum(steps)
    price = np.maximum(price, 0.1)
    base = 20000 if "BTC" in symbol else 1500 if "ETH" in symbol else 50
    close = base * price

    # 高低开收
    noise = rng.normal(loc=0, scale=0.003, size=bars)
    open_ = close * (1 + noise)
    high = np.maximum(open_, close) * (1 + np.abs(noise) * 1.5)
    low = np.minimum(open_, close) * (1 - np.abs(noise) * 1.5)
    volume = rng.integers(low=100, high=1000, size=bars)

    idx = pd.date_range(
        end=pd.Timestamp.utcnow().floor("min"),
        periods=bars,
        freq=freq,
    )
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
    return df


# ===================== 决策引擎 =====================

@dataclass
class EngineConfig:
    fast_ma: int = 10
    slow_ma: int = 40
    rsi_period: int = 14
    rsi_buy: float = 45.0
    rsi_sell: float = 60.0
    atr_period: int = 14
    edge_scale: float = 2.0  # 用于将信号压缩到 [-1,1]


class BaseEngine:
    def __init__(self, name: str, cfg: EngineConfig):
        self.name = name
        self.cfg = cfg

    def _calc_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算基础指标，并返回带指标列的 df 副本。"""
        df = df.copy()
        close = df["close"]

        # EMA 均线
        df["ma_fast"] = close.ewm(span=self.cfg.fast_ma, adjust=False).mean()
        df["ma_slow"] = close.ewm(span=self.cfg.slow_ma, adjust=False).mean()
        df["ma_diff"] = df["ma_fast"] - df["ma_slow"]

        # RSI
        delta = close.diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        roll_up = pd.Series(gain, index=df.index).rolling(self.cfg.rsi_period).mean()
        roll_down = pd.Series(loss, index=df.index).rolling(self.cfg.rsi_period).mean()
        rs = roll_up / (roll_down + 1e-8)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        # ATR
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - close.shift(1)).abs()
        low_close = (df["low"] - close.shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(self.cfg.atr_period).mean()

        return df

    def generate_edge_series(self, df: pd.DataFrame) -> pd.Series:
        """
        返回一个 [-1,1] 的 edge 序列：
        >0 代表倾向做多，<0 代表倾向做空（当前我们只用多头）。
        """
        df = self._calc_indicators(df)
        ma_norm = np.tanh(self.cfg.edge_scale * df["ma_diff"] / df["close"])
        rsi_norm = (50.0 - df["rsi"]) / 50.0  # rsi<50 倾向做多；>50 倾向做空

        edge = 0.6 * ma_norm + 0.4 * rsi_norm
        edge = edge.clip(-1.0, 1.0).fillna(0.0)
        return edge

    def direction_from_edge(self, edge_value: float) -> int:
        """基于 edge 确定方向。当前我们只做多：edge>0 才允许开多。"""
        return 1 if edge_value > 0 else 0


class BaselineEngine(BaseEngine):
    """基础版：纯技术指标。"""
    pass


class AiProdEngine(BaseEngine):
    """
    AI 版：目前仍然使用同样的指标逻辑，
    但预留位置给将来接入 ProductionTradingSystem 或大模型信号。
    """

    def generate_edge_series(self, df: pd.DataFrame) -> pd.Series:
        # 暂时：在 Baseline 的 edge 基础上做一点非线性放大，鼓励明显趋势
        base_edge = super().generate_edge_series(df)
        # 显著 edge 放大，弱 edge 压缩
        amplified = np.sign(base_edge) * (np.abs(base_edge) ** 1.2)
        return amplified.clip(-1.0, 1.0)


# ===================== 回测配置 & 记录 =====================

@dataclass
class RiskConfig:
    """
    风控 & 资金管理配置。

    所有比例都是「相对于当前净值」的百分比。
    """

    risk_per_trade: float = 0.01          # 单笔风险 1% 资金
    max_r_multiple: float = 3.0           # 单笔最大 R 倍数（限制极端暴利）
    sl_pct: float = 0.01                  # 止损距离 1%
    tp_pct: float = 0.02                  # 止盈距离 2% 基础
    trail_when_r: float = 1.0             # 当浮盈 >= 1R 时启动跟踪止损
    trail_lock_r: float = 0.5             # 启动后至少锁定 0.5R 收益

    max_dd_soft: float = 0.2              # 软回撤阈值 20%
    max_dd_hard: float = 0.4              # 硬回撤阈值 40%

    cold_streak_trades: int = 6           # 连亏 N 笔进入冷静
    cold_streak_dd: float = 0.15          # 或 DD 超过 15%

    cold_lookback_bars: int = 12          # 冷静期至少观察这么多根 K
    daily_loss_limit: float = 0.08        # 单日最大亏损 8%

    extract_trigger: float = 0.10         # 每净值新高上涨 10% 触发一次抽取
    extract_fraction: float = 0.2         # 抽取增量收益的 20% 到“保险柜”


@dataclass
class Trade:
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: int
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    r_multiple: float


@dataclass
class SymbolStats:
    trades: List[Trade]
    equity_curve: pd.Series
    banked_profit: float  # 保险柜


# ===================== 回测主体 =====================

class SmartBacktestV4:
    def __init__(
        self,
        symbols: List[str],
        days: int,
        engine_name: str = "baseline",
        initial_capital: float = 10_000.0,
        leverage: float = 3.0,
        use_real_data: bool = False,
    ):
        self.symbols = symbols
        self.days = days
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.use_real_data = use_real_data

        self.risk_cfg = RiskConfig()

        cfg = EngineConfig()
        if engine_name == "ai_prod":
            self.engine = AiProdEngine("ai_prod", cfg)
        else:
            self.engine = BaselineEngine("baseline", cfg)
        self.engine_name = engine_name

        self.results: Dict[str, SymbolStats] = {}
        self.global_trades: List[Trade] = []

    # ---------- 数据加载 ----------

    def _load_symbol_data(self, symbol: str) -> pd.DataFrame:
        if self.use_real_data:
            try:
                from real_market_data import load_market_data_for_backtest
                df = load_market_data_for_backtest(
                    symbol=symbol,
                    days=self.days,
                    interval="5m",
                )
                if df is not None and len(df) > 100:
                    logger.info(f"📊 使用真实市场数据: {symbol} ({len(df)} 行)")
                    return ensure_datetime_index(df)
                else:
                    logger.warning(f"⚠️ 真实数据不足，使用模拟数据: {symbol}")
            except Exception as e:
                logger.error(f"❌ 加载真实数据失败 ({symbol}): {e}")
        # fallback
        df = generate_synthetic_ohlcv(symbol, self.days, freq="5min")
        logger.info(f"📊 使用模拟市场数据: {symbol} ({len(df)} 行)")
        return ensure_datetime_index(df)

    # ---------- 单品种回测 ----------

    def _run_single_symbol(self, symbol: str) -> SymbolStats:
        df = self._load_symbol_data(symbol)
        edge = self.engine.generate_edge_series(df)
        df = df.copy()
        df["edge"] = edge

        capital = self.initial_capital
        equity_peak = capital
        banked = 0.0

        last_extraction_anchor = capital
        open_position = None  # (entry_price, size, side, entry_equity, entry_time, risk_amount, stop_price, tp_price, sl_pct, tp_pct)
        trades: List[Trade] = []
        equity_list = []

        cold_mode = False
        cold_bars_left = 0
        consec_losses = 0

        current_day = None
        day_start_equity = capital

        for ts, row in df.iterrows():
            price = float(row["close"])
            edge_val = float(row["edge"])
            atr = float(row.get("atr", 0.0) or 0.0)

            # 更新当前日期 & 日亏损限制
            day = ts.date()
            if current_day is None:
                current_day = day
                day_start_equity = capital
            elif day != current_day:
                # 新的一天，重置
                current_day = day
                day_start_equity = capital

            # 更新实时净值和回撤
            equity_peak = max(equity_peak, capital)
            dd = 0.0 if equity_peak <= 0 else 1.0 - capital / equity_peak

            # 动态调整 risk_per_trade
            dynamic_risk = self._dynamic_risk_per_trade(dd)

            # 冷静期逻辑
            if cold_mode:
                cold_bars_left -= 1
                if cold_bars_left <= 0 and dd < self.risk_cfg.cold_streak_dd * 0.8:
                    cold_mode = False
                    consec_losses = 0
                    logger.info(f"🧊 冷静期结束，恢复交易: {symbol} @ {ts}")
                equity_list.append(capital + banked)
                # 冷静期内不允许开新仓，但可以根据价差平已有仓
                if open_position is not None:
                    capital, open_position, closed_trade = self._check_exit(
                        ts, price, open_position, capital
                    )
                    if closed_trade is not None:
                        trades.append(closed_trade)
                        self.global_trades.append(closed_trade)
                continue

            # 检查当前持仓止损/止盈
            if open_position is not None:
                capital, open_position, closed_trade = self._check_exit(
                    ts, price, open_position, capital
                )
                if closed_trade is not None:
                    trades.append(closed_trade)
                    self.global_trades.append(closed_trade)

                    if closed_trade.pnl < 0:
                        consec_losses += 1
                    else:
                        consec_losses = 0

                    # 回撤 & 连续亏损触发冷静期
                    equity_peak = max(equity_peak, capital)
                    dd = 0.0 if equity_peak <= 0 else 1.0 - capital / equity_peak
                    if (
                        consec_losses >= self.risk_cfg.cold_streak_trades
                        or dd >= self.risk_cfg.cold_streak_dd
                    ):
                        cold_mode = True
                        cold_bars_left = self.risk_cfg.cold_lookback_bars
                        logger.info(
                            f"🧊 触发冷静期: {symbol}, 连亏={consec_losses}, DD={dd:.2%}, @ {ts}"
                        )

            # 日内亏损限制：如果当日亏损超过限制，就不再新开仓
            day_loss = (day_start_equity - capital) / max(day_start_equity, 1e-8)
            hit_daily_loss = day_loss >= self.risk_cfg.daily_loss_limit

            # 尝试开新仓（只做多）
            if open_position is None and not cold_mode and not hit_daily_loss:
                direction = self.engine.direction_from_edge(edge_val)
                if direction > 0 and edge_val > 0.2:
                    # 估算本笔风险 = risk_per_trade * equity
                    equity_now = capital
                    risk_amount = dynamic_risk * equity_now

                    # 止损距离：max(固定 SL, ATR-based)
                    sl_pct = max(self.risk_cfg.sl_pct, (atr / price) * 0.8 if atr > 0 else 0)
                    tp_pct = self.risk_cfg.tp_pct

                    # 防止 sl_pct 过小导致仓位过大
                    if sl_pct <= 0:
                        sl_pct = self.risk_cfg.sl_pct

                    # 计算名义仓位价值（保证风险不超过 risk_amount）
                    notional_at_risk = risk_amount / sl_pct
                    # 杠杆控制
                    max_notional = equity_now * self.leverage
                    notional = min(notional_at_risk, max_notional)

                    if notional > 0:
                        size = notional / price
                        entry_price = price
                        stop_price = entry_price * (1 - sl_pct)
                        # 首个 TP 按基础 tp_pct，后续有 trail 机制
                        tp_price = entry_price * (1 + tp_pct)

                        open_position = (
                            entry_price,
                            size,
                            1,  # side=多头
                            equity_now,
                            ts,
                            risk_amount,
                            stop_price,
                            tp_price,
                            sl_pct,
                            tp_pct,
                        )

            # 每根 K 结束记录总权益（含保险柜）
            equity_list.append(capital + banked)

            # 净值创新高 -> 抽取利润
            total_equity = capital + banked
            if total_equity > last_extraction_anchor * (1 + self.risk_cfg.extract_trigger):
                delta = total_equity - last_extraction_anchor
                to_bank = delta * self.risk_cfg.extract_fraction
                banked += to_bank
                capital -= to_bank
                last_extraction_anchor = capital
                logger.debug(
                    f"🏦 抽取利润: {symbol}, 抽取={to_bank:.2f}, banked={banked:.2f}, capital={capital:.2f}"
                )

        equity_curve = pd.Series(equity_list, index=df.index)
        return SymbolStats(trades=trades, equity_curve=equity_curve, banked_profit=banked)

    def _check_exit(
        self,
        ts: pd.Timestamp,
        price: float,
        pos,
        capital: float,
    ) -> Tuple[float, Optional[Tuple], Optional[Trade]]:
        """
        检查是否触发止损/止盈/跟踪止盈。
        """
        (
            entry_price,
            size,
            side,
            entry_equity,
            entry_time,
            risk_amount,
            stop_price,
            tp_price,
            sl_pct,
            tp_pct,
        ) = pos

        pnl = (price - entry_price) * size * side
        r_multiple = pnl / max(risk_amount, 1e-8)

        exit_reason = None

        # 硬限制最大 R 倍数，保护统计稳定性
        if r_multiple >= self.risk_cfg.max_r_multiple:
            exit_reason = "max_r_cap"
        # 止盈
        elif price >= tp_price:
            exit_reason = "take_profit"
        # 止损
        elif price <= stop_price:
            exit_reason = "stop_loss"
        # 简单 trailing：当浮盈超过 trail_when_r *R 时，将止损抬到锁定 trail_lock_r *R
        elif r_multiple >= self.risk_cfg.trail_when_r:
            locked_price = entry_price * (1 + self.risk_cfg.trail_lock_r * sl_pct)
            if locked_price > stop_price:
                stop_price = locked_price  # 更新止损线

        if exit_reason is None:
            # 继续持仓
            new_pos = (
                entry_price,
                size,
                side,
                entry_equity,
                entry_time,
                risk_amount,
                stop_price,
                tp_price,
                sl_pct,
                tp_pct,
            )
            return capital, new_pos, None

        # 平仓
        capital_after = capital + pnl
        trade = Trade(
            symbol="",
            entry_time=entry_time,
            exit_time=ts,
            side=side,
            entry_price=entry_price,
            exit_price=price,
            size=size,
            pnl=pnl,
            r_multiple=max(min(r_multiple, self.risk_cfg.max_r_multiple), -5.0),
        )
        return capital_after, None, trade

    def _dynamic_risk_per_trade(self, dd: float) -> float:
        """
        根据回撤调整 risk_per_trade：
        - DD < soft: 正常
        - soft <= DD < hard: 线性下降到一半
        - DD >= hard: 极限收缩到 0.25 倍
        """
        base = self.risk_cfg.risk_per_trade
        if dd <= self.risk_cfg.max_dd_soft:
            return base
        if dd >= self.risk_cfg.max_dd_hard:
            return base * 0.25
        # 线性插值 soft -> hard: 1.0 -> 0.5
        t = (dd - self.risk_cfg.max_dd_soft) / (
            self.risk_cfg.max_dd_hard - self.risk_cfg.max_dd_soft
        )
        scale = 1.0 - 0.5 * t
        return base * scale

    # ---------- 总体回测 ----------

    def run(self):
        logger.info("🚀 智能回测系统初始化完成")
        logger.info(
            f"💰 初始资金: ${self.initial_capital:,.2f}, 杠杆: {self.leverage:.1f}x, "
            f"使用真实数据: {self.use_real_data}, 引擎: {self.engine_name}"
        )
        logger.info(
            f"🎯 开始智能回测: {self.symbols}, 天数={self.days} "
        )

        for sym in self.symbols:
            logger.info(f"🔍 测试币种: {sym}")
            stats = self._run_single_symbol(sym)
            # 回填 symbol 名
            for t in stats.trades:
                t.symbol = sym
            self.results[sym] = stats

        self._report()

    # ---------- 报告 & 打分 ----------

    def _report(self):
        all_trades: List[Trade] = []
        total_banked = 0.0
        combined_equity = None

        for sym, stats in self.results.items():
            all_trades.extend(stats.trades)
            total_banked += stats.banked_profit
            if combined_equity is None:
                combined_equity = stats.equity_curve
            else:
                combined_equity = combined_equity.add(stats.equity_curve, fill_value=0.0)

        if combined_equity is None or len(combined_equity) == 0:
            logger.warning("⚠️ 没有生成任何交易，无法出报告。")
            return

        total_trades = len(all_trades)
        total_return = combined_equity.iloc[-1] - self.initial_capital
        avg_win_rate = (
            np.mean([1 if t.pnl > 0 else 0 for t in all_trades]) if total_trades > 0 else 0.0
        )

        # 最大回撤
        peak = -np.inf
        dd_list = []
        for v in combined_equity:
            peak = max(peak, v)
            dd_list.append(0 if peak <= 0 else 1.0 - v / peak)
        max_dd = max(dd_list) if dd_list else 0.0

        # 粗略年化（月化）
        months = max(self.days / 30.0, 1e-6)
        total_ret_pct = total_return / self.initial_capital
        monthly_ret = (1 + total_ret_pct) ** (1 / months) - 1
        monthly_ret_pct = monthly_ret * 100

        logger.info("")
        logger.info("=" * 80)
        logger.info("🧠 智能量化交易系统 - 回测报告")
        logger.info("=" * 80)
        logger.info("")
        logger.info("📈 智能性能汇总:")
        logger.info(f"  测试币种: {len(self.symbols)} 个")
        logger.info(f"  总交易次数: {total_trades} 笔")
        logger.info(f"  总收益: ${total_return:,.2f}")
        logger.info(f"  最终资金: ${combined_equity.iloc[-1]:,.2f}")
        logger.info(f"  平均胜率: {avg_win_rate * 100:.1f}%")
        logger.info(f"  最大回撤: {max_dd * 100:.1f}%")
        logger.info(f"  粗略年化/月化估算: 月化≈{monthly_ret_pct:.1f}% （目标≥20%）")
        logger.info("")
        logger.info("📊 各币种智能表现:")

        for sym, stats in self.results.items():
            sym_trades = stats.trades
            sym_trades_count = len(sym_trades)
            if sym_trades_count == 0:
                win_rate = 0.0
                sym_pnl = 0.0
            else:
                sym_pnl = sum(t.pnl for t in sym_trades)
                win_rate = np.mean([1 if t.pnl > 0 else 0 for t in sym_trades])

            logger.info(
                f"  🟡 {sym}: {sym_trades_count} 笔, 胜率: {win_rate * 100:.1f}%, "
                f"收益: ${sym_pnl:,.2f}, 抽取到保险柜: ${stats.banked_profit:,.2f}"
            )

        logger.info("")
        logger.info("🏦 利润抽取 + 复利模拟（简化版）:")
        logger.info(
            f"  回测期间共抽取到“保险柜”的安全利润≈${total_banked:,.2f}；"
            f"若将这些视作完全风险隔离的收益，剩余资金继续用于复利。"
        )
        logger.info("")

        score, grade, comment = self._ai_score(
            total_return=total_return,
            max_dd=max_dd,
            win_rate=avg_win_rate,
            monthly_ret=monthly_ret,
            total_trades=total_trades,
        )
        logger.info("🤖 AI 风险收益评分:")
        logger.info(
            f"  综合得分: {score:.1f} / 100, 等级: {grade}, 评语: {comment}"
        )
        logger.info("")
        logger.info("🎉 智能回测完成！")
        logger.info("=" * 80)

    def _ai_score(
        self,
        total_return: float,
        max_dd: float,
        win_rate: float,
        monthly_ret: float,
        total_trades: int,
    ) -> Tuple[float, str, str]:
        """
        非常简单粗暴的评分函数：
        - 收益越高越好
        - 回撤越小越好
        - 胜率过低会扣分
        - 交易样本太少会扣分
        """
        if total_trades < 30:
            coverage = 0.5
        elif total_trades < 200:
            coverage = 0.8
        else:
            coverage = 1.0

        # 收益分：月化 0~10% -> 0~20 分；10~50% -> 20~40；>50% 封顶 50
        m = monthly_ret
        if m <= 0:
            gain_score = 0.0
        elif m <= 0.10:
            gain_score = 20 * (m / 0.10)
        elif m <= 0.50:
            gain_score = 20 + 20 * ((m - 0.10) / 0.40)
        else:
            gain_score = 50.0

        # 回撤分：0~10% -> 30~20；10~40% -> 20~0；>40% 直接 0
        dd = max_dd
        if dd <= 0.10:
            dd_score = 30 - 10 * (dd / 0.10)  # 0% ->30; 10%->20
        elif dd <= 0.40:
            dd_score = 20 * (1 - (dd - 0.10) / 0.30)
        else:
            dd_score = 0.0
        dd_score = max(dd_score, 0.0)

        # 胜率分：30% 以下 0；30~50% -> 0~10；50~70%->10~20；>70% 封顶 20
        wr = win_rate
        if wr <= 0.30:
            wr_score = 0.0
        elif wr <= 0.50:
            wr_score = 10 * ((wr - 0.30) / 0.20)
        elif wr <= 0.70:
            wr_score = 10 + 10 * ((wr - 0.50) / 0.20)
        else:
            wr_score = 20.0

        raw_score = gain_score + dd_score + wr_score
        score = raw_score * coverage
        score = max(0.0, min(100.0, score))

        if score >= 80:
            grade = "A"
            comment = "收益-回撤表现优秀，可考虑小比例实盘观察并逐步放大仓位。"
        elif score >= 65:
            grade = "B"
            comment = "收益尚可，风险可控，适合作为组合中的一部分策略。"
        elif score >= 50:
            grade = "C"
            comment = "策略风险收益比一般，建议先小仓位或仅用作研究参考。"
        else:
            grade = "D"
            comment = "当前表现偏弱，建议继续调参或更换信号逻辑。"

        return score, grade, comment


# ===================== CLI 入口 =====================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="智能量化回测 v4")
    parser.add_argument(
        "--symbols",
        type=str,
        required=True,
        help="逗号分隔的交易对，例如 BTC/USDT,ETH/USDT",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="回测天数（用于决定样本长度）",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="baseline",
        choices=["baseline", "ai_prod"],
        help="使用的决策引擎",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10_000.0,
        help="初始资金",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=3.0,
        help="名义杠杆（仅用于控制最大头寸，不做逐仓/全仓区分）",
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="若 real_market_data 可用，则尝试加载真实K线；否则回退到模拟数据",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    backtest = SmartBacktestV4(
        symbols=symbols,
        days=args.days,
        engine_name=args.engine,
        initial_capital=args.initial_capital,
        leverage=args.leverage,
        use_real_data=args.use_real_data,
    )
    backtest.run()


if __name__ == "__main__":
    main()
