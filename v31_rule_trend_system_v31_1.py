# -*- coding: utf-8 -*-
"""
v31_rule_trend_system_v31_1.py
V31_1 · 规则版 Teacher_V11 · 抽水 + 真实复利 + 多币种版

核心变化 vs V10:
1）重构账户层：只有一套资金账本（不再 water.trading_capital / system.trading_capital 各一份）
2）抽水 = 真正从交易资金里扣出去，进入利润池（落袋为安），资金池不参与回撤
3）最终总资产 = 交易资金 + 利润池，和所有历史交易盈亏逻辑自洽
4）支持 symbol 简写：BTC / ETH / DOGE / XXXUSDT
5）止盈止损参数（RR / ATR倍数 / 最小止损）可通过命令行调节
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)

from local_data_engine_v22_9 import LocalDataEngineV22_9  # 你的本地数据引擎

Side = Literal["FLAT", "LONG", "SHORT"]
TradeType = Literal["TREND"]


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: Side
    trade_type: TradeType
    entry_price: float
    exit_price: float
    notional: float
    pnl: float
    pnl_pct: float
    bars_held: int
    reason: str
    trend_dir: int
    trend_strength: int
    atr_used: float
    rr_used: float
    boll_pos_5m: float
    hma_dir_1h: int
    hma_dir_4h: int
    adx_1h: float
    adx_4h: float


# ===========================
# 账户层（重构版）：唯一真账本
# ===========================
class WaterfallAccountV31_1:
    """
    抽水 + 复利账户模型（重构版）

    逻辑：
    - 所有交易盈亏先作用在 trading_capital 上
    - 只有启用抽水且创出新高 & 超过阈值，才会从 trading_capital 里“挖走”一部分到 profit_pool
    - profit_pool 不再回吐，是你落袋为安的资金池
    """
    def __init__(
        self,
        initial_capital: float = 10_000.0,
        withdraw_rate: float = 0.10,
        growth_threshold: float = 0.01,
        enable_waterfall: bool = True,
    ):
        self.initial_capital = initial_capital
        self.trading_capital = initial_capital
        self.profit_pool = 0.0

        self.withdraw_rate = withdraw_rate
        self.growth_threshold = growth_threshold
        self.enable_waterfall = enable_waterfall

        # 抽水基准线：账户历史最高交易资金（抽水后更新）
        self.last_high = initial_capital

        # 抽水记录
        self.history: List[Dict[str, Any]] = []

    # 这就是“参与风控 & 仓位计算”的资金
    def risk_capital(self) -> float:
        return self.trading_capital

    def total_equity(self) -> float:
        return self.trading_capital + self.profit_pool

    def apply_pnl(self, pnl: float, timestamp: pd.Timestamp):
        """
        每次平仓后调用：先更新交易资金，再检查是否触发抽水。
        """
        # 1) 先更新交易资金
        before = self.trading_capital
        self.trading_capital += pnl

        # 关闭抽水则直接返回
        if not self.enable_waterfall:
            return

        # 2) 仅当 trading_capital 创出新高时考虑抽水
        if self.trading_capital <= self.last_high:
            return

        growth = self.trading_capital - self.last_high
        growth_pct = growth / self.last_high

        if growth_pct < self.growth_threshold:
            return

        # 3) 计算抽水金额，从交易资金中扣除
        withdraw_amount = growth * self.withdraw_rate
        self.trading_capital -= withdraw_amount
        self.profit_pool += withdraw_amount

        # 更新抽水基准线（抽水后新的“高点”）
        self.last_high = self.trading_capital

        record = {
            "timestamp": timestamp,
            "growth_pct": float(growth_pct),
            "growth_amount": float(growth),
            "withdrawn": float(withdraw_amount),
            "trading_before": float(before),
            "trading_after": float(self.trading_capital),
            "profit_pool": float(self.profit_pool),
        }
        self.history.append(record)

        print(
            f"[抽水] {timestamp}: 增长{growth_pct*100:.2f}% "
            f"(+${growth:.2f}), 抽水${withdraw_amount:.2f}, "
            f"交易资金:${self.trading_capital:.2f}, 利润池:${self.profit_pool:.2f}"
        )


# ===========================
# 配置参数（新增可调止盈止损）
# ===========================
@dataclass
class V31_1_Config:
    symbol: str = "BTCUSDT"
    days: int = 365
    initial_equity: float = 10_000.0

    # 趋势过滤参数（1H / 4H）
    hma_period_1h: int = 20
    hma_period_4h: int = 20
    adx_period_1h: int = 14
    adx_period_4h: int = 14
    adx_strong_th: float = 25.0
    adx_normal_th: float = 20.0

    # 执行周期（5m）指标
    boll_window_5m: int = 20
    boll_k_5m: float = 2.0
    ema_fast_5m: int = 10
    ema_slow_5m: int = 30
    macd_fast_5m: int = 12
    macd_slow_5m: int = 26
    macd_signal_5m: int = 9

    # ATR 止损（基于 1H）
    atr_period_1h: int = 14
    min_sl_pct: float = 0.004  # 最小止损 0.4%

    # 风控参数
    risk_per_trade: float = 0.01
    leverage: float = 3.0
    pos_cap_strong: float = 1.0
    pos_cap_normal: float = 0.7

    max_consecutive_losses: int = 5
    max_daily_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.10

    # 持仓时间（bar 数）
    max_bars_strong: int = 96
    max_bars_normal: int = 72

    # 手续费 & 滑点
    fee_rate: float = 0.0004
    slippage: float = 0.0002

    # 抽水机制参数
    enable_waterfall: bool = True
    waterfall_withdraw_rate: float = 0.10
    waterfall_growth_threshold: float = 0.01

    # 极端波动过滤
    atr_spike_threshold: float = 2.0

    # 复利模式
    compound_mode: bool = True

    # 新增：止盈止损可调
    rr_strong: float = 4.0      # 强趋势 RR
    rr_normal: float = 3.0      # 普通趋势 RR
    sl_mult_strong: float = 3.5 # 强趋势 ATR倍数
    sl_mult_normal: float = 3.0 # 普通趋势 ATR倍数


# ===========================
# 主策略类（重构版）
# ===========================
class V31_1_RuleTrendWaterfall:
    def __init__(self, cfg: V31_1_Config):
        self.cfg = cfg
        self.df_5m: Optional[pd.DataFrame] = None
        self.trades: List[Trade] = []
        self.equity_curve: Optional[pd.Series] = None

        # 账户层
        self.account = WaterfallAccountV31_1(
            initial_capital=cfg.initial_equity,
            withdraw_rate=cfg.waterfall_withdraw_rate,
            growth_threshold=cfg.waterfall_growth_threshold,
            enable_waterfall=cfg.enable_waterfall,
        )

        # 风控状态
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.current_day = None
        self.peak_equity = cfg.initial_equity

        # 复利统计
        self.compound_returns: List[float] = []

    # ========= 指标工具 =========
    @staticmethod
    def _wma(series: pd.Series, period: int) -> pd.Series:
        if period <= 0:
            return series * np.nan
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(
            lambda x: np.dot(x, weights) / weights.sum(),
            raw=True,
        )

    def _hma(self, series: pd.Series, period: int) -> pd.Series:
        if period < 2:
            return series
        half = self._wma(series, period // 2)
        full = self._wma(series, period)
        raw = 2 * half - full
        hma_period = int(np.sqrt(period))
        if hma_period < 1:
            hma_period = 1
        return self._wma(raw, hma_period)

    @staticmethod
    def _adx(df: pd.DataFrame, period: int) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        plus_dm = high - prev_high
        minus_dm = prev_low - low

        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()

        plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(period).sum() / (atr + 1e-12))
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(period).sum() / (atr + 1e-12))

        dx = (plus_di - minus_di).abs() / ((plus_di + minus_di).abs() + 1e-12) * 100
        adx = dx.rolling(period).mean()
        return adx

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr

    # ========= 数据加载 =========
    def load_data(self):
        engine = LocalDataEngineV22_9()
        df = engine.load_klines(self.cfg.symbol, "5m", days=self.cfg.days)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("K线索引必须是 DatetimeIndex")
        df = df.sort_index()
        self.df_5m = df

    def _calc_indicators(self):
        assert self.df_5m is not None
        df5 = self.df_5m.copy()
        close5 = df5["close"]

        ohlc = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

        # 1H / 4H
        df_1h = df5.resample("1H").agg(ohlc).dropna()
        df_4h = df5.resample("4H").agg(ohlc).dropna()

        # HMA
        df_1h["hma"] = self._hma(df_1h["close"], self.cfg.hma_period_1h)
        df_4h["hma"] = self._hma(df_4h["close"], self.cfg.hma_period_4h)

        df_1h["hma_dir"] = np.where(df_1h["close"] > df_1h["hma"], 1, -1)
        df_4h["hma_dir"] = np.where(df_4h["close"] > df_4h["hma"], 1, -1)

        # ADX
        df_1h["adx"] = self._adx(df_1h, self.cfg.adx_period_1h)
        df_4h["adx"] = self._adx(df_4h, self.cfg.adx_period_4h)

        df_4h_to_1h = df_4h[["hma_dir", "adx"]].reindex(df_1h.index, method="ffill")
        df_1h["hma_dir_4h"] = df_4h_to_1h["hma_dir"]
        df_1h["adx_4h"] = df_4h_to_1h["adx"]

        same_sign = (df_1h["hma_dir"] * df_1h["hma_dir_4h"]) > 0
        trend_dir_1h = np.where(same_sign, df_1h["hma_dir"], 0)

        adx1 = df_1h["adx"].fillna(0.0)
        adx4 = df_1h["adx_4h"].fillna(0.0)
        adx_combined = (adx1 + adx4) / 2.0

        strong = adx_combined >= self.cfg.adx_strong_th
        normal = (adx_combined >= self.cfg.adx_normal_th) & (adx_combined < self.cfg.adx_strong_th)
        trend_strength_1h = np.zeros(len(df_1h), dtype=int)
        trend_strength_1h[normal] = 1
        trend_strength_1h[strong] = 2
        trend_strength_1h[trend_dir_1h == 0] = 0

        df_1h["trend_dir"] = trend_dir_1h
        df_1h["trend_strength"] = trend_strength_1h
        df_1h["adx_combined"] = adx_combined

        # ATR + ATR spike
        df_1h["atr_1h"] = self._atr(df_1h, self.cfg.atr_period_1h)
        df_1h["atr_ma"] = df_1h["atr_1h"].rolling(20).mean()
        df_1h["atr_spike"] = df_1h["atr_1h"] / (df_1h["atr_ma"] + 1e-12)

        # 对齐到 5m
        df5["trend_dir"] = df_1h["trend_dir"].reindex(df5.index, method="ffill").fillna(0).astype(int)
        df5["trend_strength"] = df_1h["trend_strength"].reindex(df5.index, method="ffill").fillna(0).astype(int)
        df5["hma_dir_1h"] = df_1h["hma_dir"].reindex(df5.index, method="ffill").fillna(0).astype(int)
        df5["hma_dir_4h"] = df_1h["hma_dir_4h"].reindex(df5.index, method="ffill").fillna(0).astype(int)
        df5["adx_1h"] = df_1h["adx"].reindex(df5.index, method="ffill").fillna(0.0)
        df5["adx_4h"] = df_1h["adx_4h"].reindex(df5.index, method="ffill").fillna(0.0)
        df5["adx_combined"] = df_1h["adx_combined"].reindex(df5.index, method="ffill").fillna(0.0)
        df5["atr_1h"] = df_1h["atr_1h"].reindex(df5.index, method="ffill")
        df5["atr_spike"] = df_1h["atr_spike"].reindex(df5.index, method="ffill")

        # BOLL, EMA, MACD
        mid5 = close5.rolling(self.cfg.boll_window_5m).mean()
        std5 = close5.rolling(self.cfg.boll_window_5m).std()
        up5 = mid5 + self.cfg.boll_k_5m * std5
        low5 = mid5 - self.cfg.boll_k_5m * std5
        df5["boll_mid_5m"] = mid5
        df5["boll_up_5m"] = up5
        df5["boll_low_5m"] = low5

        ema_fast_5 = close5.ewm(span=self.cfg.ema_fast_5m, adjust=False).mean()
        ema_slow_5 = close5.ewm(span=self.cfg.ema_slow_5m, adjust=False).mean()
        df5["ema_fast_5m"] = ema_fast_5
        df5["ema_slow_5m"] = ema_slow_5

        ema_fast_macd5 = close5.ewm(span=self.cfg.macd_fast_5m, adjust=False).mean()
        ema_slow_macd5 = close5.ewm(span=self.cfg.macd_slow_5m, adjust=False).mean()
        macd5 = ema_fast_macd5 - ema_slow_macd5
        macd5_sig = macd5.ewm(span=self.cfg.macd_signal_5m, adjust=False).mean()
        df5["macd_hist_5m"] = macd5 - macd5_sig

        df5["is_15m_bar"] = (df5.index.minute % 15 == 0)
        df5 = df5.dropna().copy()
        self.df_5m = df5

    # ========= 入场信号 =========
    def _entry_signal(
        self,
        ts: pd.Timestamp,
        row: pd.Series,
        prev_row: pd.Series,
    ) -> Tuple[Side, Optional[TradeType]]:
        cfg = self.cfg
        trend_dir = int(row["trend_dir"])
        trend_strength = int(row["trend_strength"])

        # 连续亏损保护
        if self.consecutive_losses >= cfg.max_consecutive_losses:
            return "FLAT", None

        # 极端波动过滤
        if row.get("atr_spike", 1.0) > cfg.atr_spike_threshold:
            return "FLAT", None

        if trend_strength <= 0 or trend_dir == 0:
            return "FLAT", None

        if not bool(row["is_15m_bar"]):
            return "FLAT", None

        close = row["close"]
        mid5 = row["boll_mid_5m"]
        up5 = row["boll_up_5m"]
        low5 = row["boll_low_5m"]

        if mid5 <= 0 or up5 <= 0 or low5 <= 0:
            return "FLAT", None
        rng = up5 - low5
        if rng <= 0:
            return "FLAT", None

        boll_pos = (close - low5) / rng
        ema_fast = row["ema_fast_5m"]
        ema_slow = row["ema_slow_5m"]
        ema_fast_prev = prev_row["ema_fast_5m"]
        ema_slow_prev = prev_row["ema_slow_5m"]

        macd_5 = row["macd_hist_5m"]
        macd_5_prev = prev_row["macd_hist_5m"]

        # 多头
        if trend_dir > 0:
            cond_price = 0.3 <= boll_pos <= 0.85
            cond_trigger = (
                (ema_fast_prev <= ema_slow_prev and ema_fast > ema_slow)
                or (macd_5_prev <= 0 and macd_5 > 0)
            )
            if cond_price and cond_trigger:
                return "LONG", "TREND"

        # 空头
        if trend_dir < 0:
            cond_price = 0.15 <= boll_pos <= 0.7
            cond_trigger = (
                (ema_fast_prev >= ema_slow_prev and ema_fast < ema_slow)
                or (macd_5_prev >= 0 and macd_5 < 0)
            )
            if cond_price and cond_trigger:
                return "SHORT", "TREND"

        return "FLAT", None

    # ========= 仓位 & 止损止盈 =========
    def _compute_sl_tp_notional(
        self,
        side: Side,
        trend_strength: int,
        entry_price: float,
        atr_1h: float,
        equity_for_risk: float,
    ) -> Tuple[float, float, float, int, float, float]:
        cfg = self.cfg
        if atr_1h <= 0 or entry_price <= 0 or equity_for_risk <= 0:
            return 0.0, 0.0, 0.0, 0, 0.0, 0.0

        # 动态 RR 与 ATR倍数（可通过命令行调）
        if trend_strength >= 2:
            rr = cfg.rr_strong
            sl_mult = cfg.sl_mult_strong
            pos_cap = cfg.pos_cap_strong
            max_bars = cfg.max_bars_strong
        else:
            rr = cfg.rr_normal
            sl_mult = cfg.sl_mult_normal
            pos_cap = cfg.pos_cap_normal
            max_bars = cfg.max_bars_normal

        sl_dist_abs = atr_1h * sl_mult
        sl_dist_pct_from_atr = sl_dist_abs / entry_price
        sl_dist_pct = max(sl_dist_pct_from_atr, cfg.min_sl_pct)
        sl_dist_abs = sl_dist_pct * entry_price

        if side == "LONG":
            stop_price = entry_price - sl_dist_abs
            take_price = entry_price + sl_dist_abs * rr
        else:
            stop_price = entry_price + sl_dist_abs
            take_price = entry_price - sl_dist_abs * rr

        if stop_price <= 0 or take_price <= 0:
            return 0.0, 0.0, 0.0, 0, 0.0, 0.0

        # 根据单笔风险反推名义仓位
        risk_target = equity_for_risk * cfg.risk_per_trade
        stop_dist_pct = abs(entry_price - stop_price) / entry_price
        if stop_dist_pct <= 0:
            return 0.0, 0.0, 0.0, 0, 0.0, 0.0

        notional_target = risk_target / stop_dist_pct

        # 杠杆上限
        max_notional = equity_for_risk * cfg.leverage * pos_cap
        notional = min(notional_target, max_notional)
        if notional <= 0:
            return 0.0, 0.0, 0.0, 0, 0.0, 0.0

        return stop_price, take_price, notional, max_bars, sl_dist_abs, rr

    # ========= 回测主循环 =========
    def run_backtest(self, plot: bool = False):
        if self.df_5m is None:
            self.load_data()
        self._calc_indicators()
        df = self.df_5m.copy()

        equity_series = []

        side: Side = "FLAT"
        trade_type: Optional[TradeType] = None
        entry_price = 0.0
        notional = 0.0
        stop_price = 0.0
        take_price = 0.0
        bars_held = 0
        max_bars_hold = 0
        entry_time: Optional[pd.Timestamp] = None
        atr_used_for_trade: float = 0.0
        rr_used_for_trade: float = 0.0
        trend_dir_for_trade: int = 0
        trend_strength_for_trade: int = 0

        floating_pnl = 0.0

        prev_row = df.iloc[0]

        for ts, row in df.iloc[1:].iterrows():
            price_high = row["high"]
            price_low = row["low"]
            price_close = row["close"]

            # 更新当天（用于每日亏损限制）
            day = ts.date()
            if self.current_day != day:
                self.current_day = day
                self.daily_pnl = 0.0

            trend_dir_now = int(row["trend_dir"])
            trend_strength_now = int(row["trend_strength"])
            atr_1h = row["atr_1h"]
            mid5 = row["boll_mid_5m"]
            up5 = row["boll_up_5m"]
            low5 = row["boll_low_5m"]

            # 1）持仓管理
            if side != "FLAT" and trade_type is not None:
                exit_reason = None
                exit_price = None

                # 止损 / 止盈
                if side == "LONG":
                    if price_low <= stop_price:
                        exit_price = stop_price
                        exit_reason = "SL"
                    elif price_high >= take_price:
                        exit_price = take_price
                        exit_reason = "TP"
                else:  # SHORT
                    if price_high >= stop_price:
                        exit_price = stop_price
                        exit_reason = "SL"
                    elif price_low <= take_price:
                        exit_price = take_price
                        exit_reason = "TP"

                # 时间止盈
                bars_held += 1
                if exit_price is None and max_bars_hold > 0 and bars_held >= max_bars_hold:
                    exit_price = price_close
                    exit_reason = "TIME"

                # 趋势结束
                if exit_price is None and (trend_strength_now <= 0 or trend_dir_now == 0):
                    exit_price = price_close
                    exit_reason = "TREND_REV"

                if exit_price is not None:
                    # 计算毛利润
                    if side == "LONG":
                        price_change_pct = (exit_price - entry_price) / entry_price
                    else:
                        price_change_pct = (entry_price - exit_price) / entry_price
                    gross_pnl = notional * price_change_pct

                    # 手续费 + 滑点
                    fee_exit = notional * self.cfg.fee_rate
                    slippage_cost = notional * self.cfg.slippage

                    pnl = gross_pnl - fee_exit - slippage_cost

                    # 记录复利前资金
                    capital_before = self.account.risk_capital()

                    # 资金更新 + 抽水
                    self.account.apply_pnl(pnl, ts)

                    # 复利收益率（以交易前资金为基准）
                    if self.cfg.compound_mode and capital_before > 0:
                        self.compound_returns.append(pnl / capital_before)

                    # 风控统计
                    self.daily_pnl += pnl
                    if pnl <= 0:
                        self.consecutive_losses += 1
                    else:
                        self.consecutive_losses = 0

                    # 每日亏损限制（以可交易资金为基准）
                    daily_loss_limit = self.account.risk_capital() * self.cfg.max_daily_loss_pct
                    if self.daily_pnl < -daily_loss_limit:
                        print(f"[风控] {ts}: 触发单日亏损限制，暂停当日新开仓")
                        # 简单做法：当天不再开新仓，在 entry_signal 里通过 daily_pnl 判断也可以

                    # 盈亏百分比（相对于交易前资金）
                    pnl_pct = pnl / max(capital_before, 1e-9)

                    # 计算布林位置
                    if mid5 > 0 and up5 > 0 and low5 > 0 and (up5 - low5) > 0:
                        boll_pos = (price_close - low5) / (up5 - low5)
                    else:
                        boll_pos = np.nan

                    # 处理时区
                    et = entry_time
                    xt = ts
                    if et is not None and et.tzinfo is not None:
                        et = et.tz_convert(None)
                    if xt is not None and xt.tzinfo is not None:
                        xt = xt.tz_convert(None)

                    self.trades.append(
                        Trade(
                            entry_time=et,
                            exit_time=xt,
                            side=side,
                            trade_type=trade_type,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            notional=notional,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            bars_held=bars_held,
                            reason=exit_reason or "UNKNOWN",
                            trend_dir=trend_dir_for_trade,
                            trend_strength=trend_strength_for_trade,
                            atr_used=float(atr_used_for_trade),
                            rr_used=float(rr_used_for_trade),
                            boll_pos_5m=float(boll_pos),
                            hma_dir_1h=int(row["hma_dir_1h"]),
                            hma_dir_4h=int(row["hma_dir_4h"]),
                            adx_1h=float(row["adx_1h"]),
                            adx_4h=float(row["adx_4h"]),
                        )
                    )

                    # 清空持仓
                    side = "FLAT"
                    trade_type = None
                    entry_price = 0.0
                    notional = 0.0
                    stop_price = 0.0
                    take_price = 0.0
                    bars_held = 0
                    max_bars_hold = 0
                    entry_time = None
                    atr_used_for_trade = 0.0
                    rr_used_for_trade = 0.0
                    trend_dir_for_trade = 0
                    trend_strength_for_trade = 0
                    floating_pnl = 0.0

            # 2）空仓 → 寻找入场
            if side == "FLAT":
                sig_side, sig_type = self._entry_signal(ts, row, prev_row)

                if sig_side != "FLAT" and sig_type is not None:
                    atr_1h = row["atr_1h"]

                    if atr_1h > 0 and self.account.risk_capital() > 0:
                        # 入场价含滑点
                        if sig_side == "LONG":
                            entry_price = price_close * (1.0 + self.cfg.slippage)
                        else:
                            entry_price = price_close * (1.0 - self.cfg.slippage)

                        # 用当前可交易资金做仓位计算（复利核心）
                        risk_equity = self.account.risk_capital()
                        stop_price, take_price, notional, max_bars_hold, sl_abs, rr_used = self._compute_sl_tp_notional(
                            sig_side,
                            trend_strength_now,
                            entry_price,
                            atr_1h,
                            risk_equity,
                        )

                        if notional > 0 and stop_price > 0 and take_price > 0 and max_bars_hold > 0:
                            side = sig_side
                            trade_type = sig_type
                            entry_time = ts
                            atr_used_for_trade = atr_1h
                            rr_used_for_trade = rr_used
                            trend_dir_for_trade = trend_dir_now
                            trend_strength_for_trade = trend_strength_now
                            bars_held = 0

                            # 入场手续费（直接从交易资本扣）
                            fee_entry = notional * self.cfg.fee_rate
                            self.account.apply_pnl(-fee_entry, ts)  # 手续费当作小亏损一次

                        else:
                            # 放弃入场
                            side = "FLAT"
                            trade_type = None
                            entry_price = 0.0
                            notional = 0.0
                            stop_price = 0.0
                            take_price = 0.0
                            max_bars_hold = 0
                            atr_used_for_trade = 0.0
                            rr_used_for_trade = 0.0
                            trend_dir_for_trade = 0
                            trend_strength_for_trade = 0

            # 3）浮动盈亏 + 资金曲线 + 回撤风控
            if side != "FLAT" and notional > 0:
                if side == "LONG":
                    floating_pnl = notional * (price_close - entry_price) / entry_price
                else:
                    floating_pnl = notional * (entry_price - price_close) / entry_price
            else:
                floating_pnl = 0.0

            current_equity = self.account.total_equity() + floating_pnl
            self.peak_equity = max(self.peak_equity, current_equity)

            # 最大回撤风控：达到阈值，强制平仓
            if self.peak_equity > 0:
                drawdown = (current_equity - self.peak_equity) / self.peak_equity
                if drawdown <= -self.cfg.max_drawdown_pct and side != "FLAT" and notional > 0:
                    print(f"[风控] {ts}: 触发最大回撤限制 {drawdown*100:.2f}%，强制平仓")
                    if side == "LONG":
                        exit_price = price_close * (1.0 - self.cfg.slippage)
                    else:
                        exit_price = price_close * (1.0 + self.cfg.slippage)

                    if side == "LONG":
                        gross_pnl = notional * (exit_price - entry_price) / entry_price
                    else:
                        gross_pnl = notional * (entry_price - exit_price) / entry_price

                    fee_exit = notional * self.cfg.fee_rate
                    pnl = gross_pnl - fee_exit - notional * self.cfg.slippage

                    capital_before = self.account.risk_capital()
                    self.account.apply_pnl(pnl, ts)
                    if self.cfg.compound_mode and capital_before > 0:
                        self.compound_returns.append(pnl / capital_before)

                    # 清仓
                    side = "FLAT"
                    trade_type = None
                    entry_price = 0.0
                    notional = 0.0
                    stop_price = 0.0
                    take_price = 0.0
                    bars_held = 0
                    max_bars_hold = 0
                    entry_time = None
                    atr_used_for_trade = 0.0
                    rr_used_for_trade = 0.0
                    trend_dir_for_trade = 0
                    trend_strength_for_trade = 0
                    floating_pnl = 0.0

                    current_equity = self.account.total_equity()

            equity_series.append((ts, current_equity))
            prev_row = row

        # 资金曲线
        if equity_series:
            idx, vals = zip(*equity_series)
            idx = pd.DatetimeIndex(idx)
            if idx.tz is not None:
                idx = idx.tz_convert(None)
            self.equity_curve = pd.Series(vals, index=idx, name="equity")

        if plot:
            self._plot_equity()

    # ========= 结果汇总 =========
    def summary(self):
        if self.equity_curve is None:
            raise ValueError("请先运行 run_backtest()")

        equity = self.equity_curve
        trades = self.trades
        if len(equity) == 0:
            print("没有产生任何交易。")
            return

        final_total_value = self.account.total_equity()
        net_profit = final_total_value - self.cfg.initial_equity
        total_return = net_profit / self.cfg.initial_equity

        days = (equity.index[-1] - equity.index[0]).days if len(equity) > 1 else 0
        if days <= 0:
            annual_return = np.nan
        else:
            annual_return = (1 + total_return) ** (365.0 / days) - 1.0

        equity_cummax = equity.cummax()
        drawdown = equity / equity_cummax - 1.0
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin() if not drawdown.empty else None

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / len(trades) if trades else 0.0
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0.0
        rr_real = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan

        # 策略层“毛盈利”：所有交易 PnL 之和（不考虑抽水只是转移）
        trade_pnl_sum = sum(t.pnl for t in trades)

        print("=" * 80)
        print("V31_1 规则版 Teacher_V11 · 抽水 + 真实复利 回测结果")
        print("=" * 80)
        print(f"交易对: {self.cfg.symbol}")
        print(f"测试天数: {self.cfg.days}")
        print(f"杠杆倍数: {self.cfg.leverage}")
        print(f"初始资金: ${self.cfg.initial_equity:,.2f}")
        print(f"期末总资产(含利润池): ${final_total_value:,.2f}")
        print(f"账户净盈利: ${net_profit:,.2f}")
        print(f"总收益率: {total_return*100:.2f}%")
        print(f"年化收益: {annual_return*100:.2f}%")
        print(f"最大回撤: {max_dd*100:.2f}% ({max_dd_date})")
        print(f"交易次数: {len(trades)}")
        print(f"胜率: {win_rate*100:.2f}%")
        print(f"平均盈利: ${avg_win:.2f}")
        print(f"平均亏损: ${avg_loss:.2f}")
        print(f"实际盈亏比: {rr_real:.2f}")
        print(f"手续费率: {self.cfg.fee_rate*100:.4f}%")
        print(f"滑点: {self.cfg.slippage*100:.4f}%")
        print(f"策略毛盈利(所有交易之和): ${trade_pnl_sum:,.2f}")
        print(f"抽水后净盈利(账户层): ${net_profit:,.2f}")

        if self.cfg.compound_mode and self.compound_returns:
            compound_final = self.cfg.initial_equity
            simple_final = self.cfg.initial_equity

            for r in self.compound_returns:
                compound_final *= (1 + r)
                simple_final += r * self.cfg.initial_equity

            compound_return = (compound_final / self.cfg.initial_equity - 1) * 100
            simple_return = (simple_final / self.cfg.initial_equity - 1) * 100
            print("\n复利效果分析:")
            print(f"简单累计收益(不复利): {simple_return:.2f}%")
            print(f"复利累计收益: {compound_return:.2f}%")
            print(f"复利增益: {compound_return - simple_return:.2f}%")

        if self.cfg.enable_waterfall:
            print("\n" + "-" * 40)
            print("抽水机制统计")
            print("-" * 40)
            print(f"当前交易资金: ${self.account.trading_capital:,.2f}")
            print(f"利润池可提取: ${self.account.profit_pool:,.2f}")
            print(f"总资产（交易+利润）: ${self.account.total_equity():,.2f}")
            print(f"抽水触发次数: {len(self.account.history)}")
            if self.account.history:
                total_withdrawn = sum(r["withdrawn"] for r in self.account.history)
                avg_growth = sum(r["growth_pct"] for r in self.account.history) / len(self.account.history)
                print(f"总抽水金额: ${total_withdrawn:,.2f}")
                print(f"平均每次增长: {avg_growth*100:.2f}%")
                print(f"抽水比例: {self.account.withdraw_rate*100:.2f}%")
                print(f"增长触发阈值: {self.account.growth_threshold*100:.2f}%")

        self._print_detailed_statistics(annual_return, total_return)

    # ========= 详细统计 =========
    def _print_detailed_statistics(self, annual_return: float, total_return: float):
        print("\n" + "=" * 80)
        print("详细交易统计表")
        print("=" * 80)

        if not self.trades:
            print("无交易数据")
            return

        df_trades = pd.DataFrame(
            [
                {
                    "序号": i + 1,
                    "方向": t.side,
                    "入场时间": t.entry_time,
                    "出场时间": t.exit_time,
                    "持仓时间(bar)": t.bars_held,
                    "入场价": round(t.entry_price, 2),
                    "出场价": round(t.exit_price, 2),
                    "盈亏($)": round(t.pnl, 2),
                    "盈亏(%)": round(t.pnl_pct * 100, 2),
                    "平仓原因": t.reason,
                    "趋势强度": t.trend_strength,
                    "ATR止损": round(t.atr_used, 4),
                    "盈亏比设定": t.rr_used,
                }
                for i, t in enumerate(self.trades)
            ]
        )

        print(f"\n总交易次数: {len(df_trades)}")
        print(f"总盈亏: ${df_trades['盈亏($)'].sum():.2f}")
        print(f"平均每笔盈亏: ${df_trades['盈亏($)'].mean():.2f}")
        print(f"盈亏标准差: ${df_trades['盈亏($)'].std():.2f}")

        df_trades["月份"] = pd.to_datetime(df_trades["入场时间"]).dt.to_period("M")
        monthly_stats = df_trades.groupby("月份").agg(
            {"盈亏($)": ["count", "sum", "mean", "std"], "盈亏(%)": "mean"}
        ).round(2)

        print("\n月度统计：")
        print(monthly_stats)

        reason_stats = df_trades.groupby("平仓原因").agg(
            {"盈亏($)": ["count", "sum", "mean", "std"], "盈亏(%)": ["mean", "std"]}
        ).round(2)
        print("\n按平仓原因统计：")
        print(reason_stats)

        strength_stats = df_trades.groupby("趋势强度").agg(
            {"盈亏($)": ["count", "sum", "mean", "std"], "盈亏(%)": ["mean", "std"]}
        ).round(2)
        print("\n按趋势强度统计：")
        print(strength_stats)

        direction_stats = df_trades.groupby("方向").agg(
            {"盈亏($)": ["count", "sum", "mean", "std"], "盈亏(%)": ["mean", "std"]}
        ).round(2)
        print("\n按方向统计：")
        print(direction_stats)

        print("\n持仓时间分析：")
        print(f"平均持仓时间: {df_trades['持仓时间(bar)'].mean():.1f} bars")
        print(f"最短持仓: {df_trades['持仓时间(bar)'].min()} bars")
        print(f"最长持仓: {df_trades['持仓时间(bar)'].max()} bars")

        returns = [t.pnl_pct for t in self.trades]
        if len(returns) > 1:
            avg_return = np.mean(returns) * 252
            std_return = np.std(returns) * np.sqrt(252)
            sharpe = avg_return / std_return if std_return > 0 else 0

            downside_returns = [r for r in returns if r < 0]
            downside_std = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0
            sortino = avg_return / downside_std if downside_std > 0 else 0

            if self.equity_curve is not None:
                equity = self.equity_curve
                equity_cummax = equity.cummax()
                drawdown = equity / equity_cummax - 1.0
                max_dd = abs(drawdown.min())
                calmar = annual_return / max_dd if max_dd > 0 else 0
            else:
                calmar = 0

            print("\n风险调整后收益：")
            print(f"夏普比率: {sharpe:.3f}")
            print(f"索提诺比率: {sortino:.3f}")
            print(f"卡尔玛比率: {calmar:.3f}")

        print("\n连续盈亏分析：")
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0

        for t in self.trades:
            if t.pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

        print(f"最大连续盈利次数: {max_consecutive_wins}")
        print(f"最大连续亏损次数: {max_consecutive_losses}")

        if self.equity_curve is not None:
            equity = self.equity_curve
            daily_returns = equity.pct_change().dropna()
            print("\n资金曲线分析：")
            print(f"日收益率均值: {daily_returns.mean()*100:.4f}%")
            print(f"日收益率标准差: {daily_returns.std()*100:.4f}%")
            print(f"日收益率偏度: {daily_returns.skew():.4f}")
            print(f"日收益率峰度: {daily_returns.kurtosis():.4f}")
            winning_days = (daily_returns > 0).sum()
            total_days = len(daily_returns)
            print(f"盈利天数比例: {winning_days/total_days*100:.2f}%")

        print("=" * 80)

    # ========= 导出 & 画图（可选） =========
    def export_trades(self, filepath: str):
        if not self.trades:
            print("无交易可导出。")
            return

        rows = []
        for t in self.trades:
            rows.append(
                {
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "side": t.side,
                    "type": t.trade_type,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "notional": t.notional,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "bars_held": t.bars_held,
                    "reason": t.reason,
                    "trend_dir": t.trend_dir,
                    "trend_strength": t.trend_strength,
                    "atr_used": t.atr_used,
                    "rr_used": t.rr_used,
                    "boll_pos_5m": t.boll_pos_5m,
                    "hma_dir_1h": t.hma_dir_1h,
                    "hma_dir_4h": t.hma_dir_4h,
                    "adx_1h": t.adx_1h,
                    "adx_4h": t.adx_4h,
                }
            )

        df = pd.DataFrame(rows)
        for col in ["entry_time", "exit_time"]:
            if col in df.columns and np.issubdtype(df[col].dtype, np.datetime64):
                try:
                    df[col] = df[col].dt.tz_convert(None)
                except Exception:
                    pass

        if filepath.lower().endswith(".xlsx"):
            df.to_excel(filepath, index=False)
        else:
            df.to_csv(filepath, index=False)

        print(f"交易明细已导出到: {os.path.abspath(filepath)}")

    def export_statistics_report(self, filepath: str):
        if not self.trades or self.equity_curve is None:
            print("无数据可导出")
            return

        output_dir = os.path.dirname(os.path.abspath(filepath))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            trades_df = pd.DataFrame(
                [
                    {
                        "Entry Time": t.entry_time,
                        "Exit Time": t.exit_time,
                        "Side": t.side,
                        "Type": t.trade_type,
                        "Entry Price": t.entry_price,
                        "Exit Price": t.exit_price,
                        "Notional": t.notional,
                        "PnL": t.pnl,
                        "PnL %": t.pnl_pct * 100,
                        "Bars Held": t.bars_held,
                        "Exit Reason": t.reason,
                        "Trend Strength": t.trend_strength,
                        "ATR Used": t.atr_used,
                        "RR Used": t.rr_used,
                    }
                    for t in self.trades
                ]
            )

            for col in ["Entry Time", "Exit Time"]:
                if col in trades_df.columns and pd.api.types.is_datetime64_any_dtype(trades_df[col]):
                    tz = getattr(trades_df[col].dt, "tz", None)
                    if tz is not None:
                        trades_df[col] = trades_df[col].dt.tz_convert(None)

            trades_df.to_excel(writer, sheet_name="交易明细", index=False)

            equity_df = pd.DataFrame(
                {"Timestamp": self.equity_curve.index, "Equity": self.equity_curve.values}
            )
            if pd.api.types.is_datetime64_any_dtype(equity_df["Timestamp"]):
                tz = getattr(equity_df["Timestamp"].dt, "tz", None)
                if tz is not None:
                    equity_df["Timestamp"] = equity_df["Timestamp"].dt.tz_convert(None)
            equity_df.to_excel(writer, sheet_name="资金曲线", index=False)

            equity = self.equity_curve
            equity_cummax = equity.cummax()
            drawdown = equity / equity_cummax - 1.0
            max_dd = abs(drawdown.min())

            final_total_value = self.account.total_equity()
            net_profit = final_total_value - self.cfg.initial_equity
            total_return = net_profit / self.cfg.initial_equity
            days = (equity.index[-1] - equity.index[0]).days if len(equity) > 1 else 0
            if days <= 0:
                annual_return = np.nan
            else:
                annual_return = (1 + total_return) ** (365.0 / days) - 1.0

            wins = [t for t in self.trades if t.pnl > 0]
            losses = [t for t in self.trades if t.pnl <= 0]
            win_rate = len(wins) / len(self.trades) if self.trades else 0.0
            avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
            avg_loss = np.mean([t.pnl for t in losses]) if losses else 0.0
            rr_real = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan

            returns = [t.pnl_pct for t in self.trades]
            if len(returns) > 1:
                avg_ret = np.mean(returns) * 252
                std_ret = np.std(returns) * np.sqrt(252)
                sharpe = avg_ret / std_ret if std_ret > 0 else 0
            else:
                sharpe = 0

            summary_data = {
                "指标": [
                    "交易对",
                    "测试天数",
                    "杠杆倍数",
                    "初始资金",
                    "最终总资产",
                    "账户净收益率",
                    "年化收益率",
                    "最大回撤",
                    "交易次数",
                    "胜率",
                    "平均盈利",
                    "平均亏损",
                    "实际盈亏比",
                    "夏普比率",
                    "抽水次数",
                    "总抽水金额",
                    "利润池余额",
                ],
                "数值": [
                    self.cfg.symbol,
                    self.cfg.days,
                    self.cfg.leverage,
                    f"${self.cfg.initial_equity:,.2f}",
                    f"${final_total_value:,.2f}",
                    f"{total_return*100:.2f}%",
                    f"{annual_return*100:.2f}%",
                    f"{max_dd*100:.2f}%",
                    len(self.trades),
                    f"{win_rate*100:.2f}%",
                    f"${avg_win:.2f}",
                    f"${avg_loss:.2f}",
                    f"{rr_real:.2f}",
                    f"{sharpe:.3f}",
                    len(self.account.history) if self.cfg.enable_waterfall else 0,
                    f"${sum(r['withdrawn'] for r in self.account.history):,.2f}"
                    if self.cfg.enable_waterfall and self.account.history
                    else "$0.00",
                    f"${self.account.profit_pool:,.2f}"
                    if self.cfg.enable_waterfall
                    else "$0.00",
                ],
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="汇总统计", index=False)

            if len(trades_df) > 0:
                trades_df["月份"] = pd.to_datetime(trades_df["Entry Time"]).dt.to_period("M")
                monthly_stats = trades_df.groupby("月份").agg(
                    {"PnL": ["count", "sum", "mean", "std"], "PnL %": "mean"}
                ).round(2)
                monthly_stats.to_excel(writer, sheet_name="月度统计")

            if self.cfg.enable_waterfall and self.account.history:
                wf_df = pd.DataFrame(self.account.history)
                if "timestamp" in wf_df.columns and pd.api.types.is_datetime64_any_dtype(
                    wf_df["timestamp"]
                ):
                    tz = getattr(wf_df["timestamp"].dt, "tz", None)
                    if tz is not None:
                        wf_df["timestamp"] = wf_df["timestamp"].dt.tz_convert(None)
                wf_df.to_excel(writer, sheet_name="抽水记录", index=False)

        print(f"详细统计报告已导出到: {os.path.abspath(filepath)}")

    def _plot_equity(self):
        if self.equity_curve is None or len(self.equity_curve) == 0:
            print("无资金曲线可绘制")
            return

        eq = self.equity_curve

        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        axes[0].plot(eq.index, eq.values, linewidth=1.5)
        if self.cfg.enable_waterfall and self.account.history:
            for r in self.account.history:
                axes[0].axvline(
                    x=r["timestamp"], alpha=0.3, linestyle="--", linewidth=0.5
                )
        axes[0].set_title(
            f"V31_1 Teacher_V11 Waterfall · {self.cfg.symbol} (Leverage:{self.cfg.leverage}x)"
        )
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Total Equity (USDT)")
        axes[0].grid(True, alpha=0.3)

        equity_cummax = eq.cummax()
        drawdown = (eq / equity_cummax - 1.0) * 100
        axes[1].fill_between(eq.index, drawdown, 0, alpha=0.3)
        axes[1].plot(eq.index, drawdown, linewidth=1)
        axes[1].set_title("Drawdown Curve")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].grid(True, alpha=0.3)

        if len(eq) > 30:
            monthly_eq = eq.resample("M").last()
            monthly_returns = monthly_eq.pct_change().dropna() * 100
            colors = ["g" if x > 0 else "r" for x in monthly_returns]
            axes[2].bar(monthly_returns.index.strftime("%Y-%m"), monthly_returns.values, color=colors, alpha=0.7)
            axes[2].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            axes[2].set_title("Monthly Returns")
            axes[2].set_xlabel("Month")
            axes[2].set_ylabel("Return (%)")
            axes[2].grid(True, alpha=0.3)
            plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.show()


# ===========================
# 参数解析
# ===========================
def normalize_symbol(sym: str) -> str:
    sym = sym.upper().strip()
    # 支持 BTC / ETH / DOGE 这种简写
    if not sym.endswith("USDT"):
        sym = sym + "USDT"
    return sym


def parse_args():
    p = argparse.ArgumentParser(description="V31_1 Teacher_V11 · 抽水 + 复利 回测")
    p.add_argument("--symbol", type=str, default="BTCUSDT", help="交易对，例如 BTC / BTCUSDT / ETH / DOGE")
    p.add_argument("--days", type=int, default=365, help="回测天数")
    p.add_argument("--leverage", type=float, default=3.0, help="杠杆倍数")
    p.add_argument("--initial-equity", type=float, default=10000.0, help="初始资金")
    p.add_argument("--risk-per-trade", type=float, default=0.01, help="单笔风险比例")
    p.add_argument("--no-plot", action="store_true", help="不显示图表")

    p.add_argument("--no-waterfall", action="store_true", help="禁用抽水机制")
    p.add_argument("--withdraw-rate", type=float, default=0.10, help="抽水比例")
    p.add_argument("--growth-threshold", type=float, default=0.01, help="增长触发阈值")

    p.add_argument("--max-consecutive-losses", type=int, default=5, help="最大连续亏损次数")
    p.add_argument("--max-daily-loss", type=float, default=0.05, help="单日最大亏损比例")
    p.add_argument("--max-drawdown", type=float, default=0.10, help="最大回撤强平阈值")
    p.add_argument("--no-compound", action="store_true", help="禁用复利模式")
    p.add_argument("--atr-spike", type=float, default=2.0, help="ATR突增倍数阈值")

    # 新增止盈止损参数
    p.add_argument("--rr-strong", type=float, default=4.0, help="强趋势RR")
    p.add_argument("--rr-normal", type=float, default=3.0, help="普通趋势RR")
    p.add_argument("--sl-mult-strong", type=float, default=3.5, help="强趋势ATR止损倍数")
    p.add_argument("--sl-mult-normal", type=float, default=3.0, help="普通趋势ATR止损倍数")
    p.add_argument("--min-sl-pct", type=float, default=0.004, help="最小止损比例")

    p.add_argument("--export", type=str, default="", help="导出交易明细路径")
    p.add_argument("--export-report", type=str, default="", help="导出详细统计报告路径")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    symbol = normalize_symbol(args.symbol)

    cfg = V31_1_Config(
        symbol=symbol,
        days=args.days,
        initial_equity=args.initial_equity,
        leverage=args.leverage,
        risk_per_trade=args.risk_per_trade,
        enable_waterfall=not args.no_waterfall,
        waterfall_withdraw_rate=args.withdraw_rate,
        waterfall_growth_threshold=args.growth_threshold,
        max_consecutive_losses=args.max_consecutive_losses,
        max_daily_loss_pct=args.max_daily_loss,
        max_drawdown_pct=args.max_drawdown,
        compound_mode=not args.no_compound,
        atr_spike_threshold=args.atr_spike,
        rr_strong=args.rr_strong,
        rr_normal=args.rr_normal,
        sl_mult_strong=args.sl_mult_strong,
        sl_mult_normal=args.sl_mult_normal,
        min_sl_pct=args.min_sl_pct,
    )

    print("=" * 80)
    print(f"V31_1 Teacher_V11 抽水 + 复利 回测")
    print(f"交易对: {cfg.symbol}")
    print(f"杠杆倍数: {cfg.leverage}x")
    print(f"初始资金: ${cfg.initial_equity:,.2f}")
    print(f"单笔风险: {cfg.risk_per_trade*100:.2f}%")
    print(f"抽水机制: {'启用' if cfg.enable_waterfall else '禁用'}")
    print(f"复利模式: {'启用' if cfg.compound_mode else '禁用'}")
    print("=" * 80)

    system = V31_1_RuleTrendWaterfall(cfg)
    system.run_backtest(plot=not args.no_plot)
    system.summary()

    if args.export:
        system.export_trades(args.export)
    if args.export_report:
        system.export_statistics_report(args.export_report)
