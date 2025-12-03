# v31_rule_trend_system_v9_waterfall_fixed.py
# V31 · 规则版 Teacher_V9 · 抽水机制版（修正版）
# 修复了拼写错误，并简化了抽水逻辑

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from local_data_engine_v22_9 import LocalDataEngineV22_9


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


class ProfitWaterfall:
    """利润抽水与资金池管理（简化版）"""
    
    def __init__(self, 
                 initial_capital: float = 10_000.0,
                 withdraw_rate: float = 0.10,      # 抽水比例10%
                 growth_threshold: float = 0.01):  # 1%增长触发
        
        self.initial_capital = initial_capital
        self.trading_capital = initial_capital     # 用于交易的本金
        self.profit_pool = 0.0                     # 利润池（可提取）
        self.withdraw_rate = withdraw_rate
        self.growth_threshold = growth_threshold
        
        # 抽水基准线（上一次抽水时的总资产）
        self.last_waterfall_level = initial_capital  # 这里修正了拼写
        
        # 记录历史
        self.history = []
    
    def get_total_value(self) -> float:
        """获取总资产（交易资金 + 利润池）"""
        return self.trading_capital + self.profit_pool
    
    def update(self, current_total_value: float, timestamp: pd.Timestamp) -> float:
        """
        检查是否需要抽水，返回更新后的交易资金
        """
        # 计算相对于上一次抽水基准的增长百分比
        growth_pct = current_total_value / self.last_waterfall_level - 1.0  # 这里修正了拼写
        
        # 如果增长达到阈值
        if growth_pct >= self.growth_threshold:
            # 计算增长金额
            growth_amount = current_total_value - self.last_waterfall_level  # 这里修正了拼写
            
            # 计算抽水金额（10%）
            withdraw_amount = growth_amount * self.withdraw_rate
            
            # 剩余增长部分加入交易资金
            remaining_growth = growth_amount - withdraw_amount
            
            # 更新资金池
            self.profit_pool += withdraw_amount
            
            # 更新交易资金
            self.trading_capital += remaining_growth
            
            # 记录历史
            record = {
                'timestamp': timestamp,
                'growth_pct': growth_pct,
                'growth_amount': growth_amount,
                'withdrawn': withdraw_amount,
                'remaining_growth': remaining_growth,
                'new_trading_capital': self.trading_capital,
                'new_profit_pool': self.profit_pool
            }
            self.history.append(record)
            
            # 更新抽水基准线
            self.last_waterfall_level = current_total_value  # 这里修正了拼写
            
            # 输出日志
            print(f"[抽水] {timestamp}: 增长{growth_pct*100:.2f}% (+${growth_amount:.2f}), "
                  f"抽水${withdraw_amount:.2f}, 交易资金:${self.trading_capital:.2f}, "
                  f"利润池:${self.profit_pool:.2f}")
        
        return self.trading_capital


@dataclass
class V31V9Config:
    symbol: str = "BTCUSDT"
    days: int = 365
    initial_equity: float = 10_000.0

    # 趋势过滤参数（1H / 4H）
    hma_period_1h: int = 20
    hma_period_4h: int = 20
    adx_period_1h: int = 14
    adx_period_4h: int = 14
    adx_strong_th: float = 25.0   # 强趋势阈值
    adx_normal_th: float = 20.0   # 中等趋势阈值

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
    risk_per_trade: float = 0.01  # 单笔风险 1%
    leverage: float = 3.0  # 在这里修改杠杆值！比如改成2.0或5.0
    pos_cap_strong: float = 1.0   # 强趋势名义仓位上限（占杠杆后资金）
    pos_cap_normal: float = 0.7   # 中等趋势仓位上限

    # 持仓时间限制（以 5m bar 计）
    max_bars_strong: int = 96    # ≈8 小时
    max_bars_normal: int = 72    # ≈6 小时

    # 手续费 & 滑点
    fee_rate: float = 0.0004  # 手续费率0.04%
    slippage: float = 0.0002  # 滑点0.02%
    
    # 抽水机制参数
    enable_waterfall: bool = True               # 是否启用抽水机制
    waterfall_withdraw_rate: float = 0.10       # 抽水比例10%
    waterfall_growth_threshold: float = 0.01    # 增长1%触发


class V31RuleTrendSystemV9Waterfall:
    def __init__(self, cfg: V31V9Config):
        self.cfg = cfg
        self.df_5m: Optional[pd.DataFrame] = None
        self.trades: List[Trade] = []
        self.equity_curve: Optional[pd.Series] = None
        
        # 抽水机制
        self.waterfall = ProfitWaterfall(
            initial_capital=cfg.initial_equity,
            withdraw_rate=cfg.waterfall_withdraw_rate,
            growth_threshold=cfg.waterfall_growth_threshold
        )
        
        # 资金状态
        self.trading_capital = cfg.initial_equity
        self.total_value = cfg.initial_equity
        self.floating_pnl = 0.0

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

    # ========= 数据加载与指标计算 =========
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

        # ---- 1H & 4H 数据 ----
        df_1h = df5.resample("1H").agg(ohlc).dropna()
        df_4h = df5.resample("4H").agg(ohlc).dropna()

        # ---- HMA ----
        df_1h["hma"] = self._hma(df_1h["close"], self.cfg.hma_period_1h)
        df_4h["hma"] = self._hma(df_4h["close"], self.cfg.hma_period_4h)

        df_1h["hma_dir"] = np.where(df_1h["close"] > df_1h["hma"], 1, -1)
        df_4h["hma_dir"] = np.where(df_4h["close"] > df_4h["hma"], 1, -1)

        # ---- ADX ----
        df_1h["adx"] = self._adx(df_1h, self.cfg.adx_period_1h)
        df_4h["adx"] = self._adx(df_4h, self.cfg.adx_period_4h)

        # 对齐 4H 指标到 1H
        df_4h_to_1h = df_4h[["hma_dir", "adx"]].reindex(df_1h.index, method="ffill")
        df_1h["hma_dir_4h"] = df_4h_to_1h["hma_dir"]
        df_1h["adx_4h"] = df_4h_to_1h["adx"]

        # 趋势方向：1H 与 4H 同向才有效
        same_sign = (df_1h["hma_dir"] * df_1h["hma_dir_4h"]) > 0
        trend_dir_1h = np.where(same_sign, df_1h["hma_dir"], 0)

        # 趋势强度：结合 1H + 4H ADX
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

        # ---- ATR (1H) ----
        df_1h["atr_1h"] = self._atr(df_1h, self.cfg.atr_period_1h)

        # ---- 对齐所有趋势字段到 5m ----
        df5["trend_dir"] = df_1h["trend_dir"].reindex(df5.index, method="ffill").fillna(0).astype(int)
        df5["trend_strength"] = df_1h["trend_strength"].reindex(df5.index, method="ffill").fillna(0).astype(int)
        df5["hma_dir_1h"] = df_1h["hma_dir"].reindex(df5.index, method="ffill").fillna(0).astype(int)
        df5["hma_dir_4h"] = df_1h["hma_dir_4h"].reindex(df5.index, method="ffill").fillna(0).astype(int)
        df5["adx_1h"] = df_1h["adx"].reindex(df5.index, method="ffill").fillna(0.0)
        df5["adx_4h"] = df_1h["adx_4h"].reindex(df5.index, method="ffill").fillna(0.0)
        df5["adx_combined"] = df_1h["adx_combined"].reindex(df5.index, method="ffill").fillna(0.0)
        df5["atr_1h"] = df_1h["atr_1h"].reindex(df5.index, method="ffill")

        # ---- 5m BOLL, EMA, MACD ----
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
        macd5_hist = macd5 - macd5_sig
        df5["macd_hist_5m"] = macd5_hist

        # 标记 15m bar 起点（用于控制入场频率）
        df5["is_15m_bar"] = (df5.index.minute % 15 == 0)

        # 丢弃前期 NaN
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

        # === 多头 ===
        if trend_dir > 0:
            cond_price = 0.3 <= boll_pos <= 0.85
            cond_trigger = (
                (ema_fast_prev <= ema_slow_prev and ema_fast > ema_slow)
                or (macd_5_prev <= 0 and macd_5 > 0)
            )
            if cond_price and cond_trigger:
                return "LONG", "TREND"

        # === 空头 ===
        if trend_dir < 0:
            cond_price = 0.15 <= boll_pos <= 0.7
            cond_trigger = (
                (ema_fast_prev >= ema_slow_prev and ema_fast < ema_slow)
                or (macd_5_prev >= 0 and macd_5 < 0)
            )
            if cond_price and cond_trigger:
                return "SHORT", "TREND"

        return "FLAT", None

    # ========= 仓位 + 止损/止盈 =========
    def _compute_sl_tp_notional(
        self,
        side: Side,
        trend_strength: int,
        entry_price: float,
        atr_1h: float,
        equity: float,
    ) -> Tuple[float, float, float, int, float, float]:
        cfg = self.cfg
        if atr_1h <= 0 or entry_price <= 0 or equity <= 0:
            return 0.0, 0.0, 0.0, 0, 0.0, 0.0

        # 动态 RR + ATR 止损倍数
        if trend_strength >= 2:
            rr = 4.0
            sl_mult = 3.5
            pos_cap = cfg.pos_cap_strong
            max_bars = cfg.max_bars_strong
        else:
            rr = 3.0
            sl_mult = 3.0
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

        # 风险预算反推名义仓位
        risk_target = equity * cfg.risk_per_trade
        stop_dist_pct = abs(entry_price - stop_price) / entry_price
        if stop_dist_pct <= 0:
            return 0.0, 0.0, 0.0, 0, 0.0, 0.0

        notional_target = risk_target / stop_dist_pct

        # 这里使用杠杆计算最大名义仓位
        max_notional = equity * cfg.leverage * pos_cap
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

        # 初始化资金
        self.trading_capital = self.cfg.initial_equity
        self.total_value = self.cfg.initial_equity
        self.floating_pnl = 0.0
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

        prev_row = df.iloc[0]

        for ts, row in df.iloc[1:].iterrows():
            price_high = row["high"]
            price_low = row["low"]
            price_close = row["close"]

            trend_dir_now = int(row["trend_dir"])
            trend_strength_now = int(row["trend_strength"])
            atr_1h = row["atr_1h"]
            mid5 = row["boll_mid_5m"]
            up5 = row["boll_up_5m"]
            low5 = row["boll_low_5m"]

            # === 1) 抽水机制检查 ===
            if self.cfg.enable_waterfall:
                # 计算当前总资产
                if side != "FLAT":
                    # 有持仓时，计算浮动盈亏
                    if side == "LONG":
                        self.floating_pnl = notional * (price_close - entry_price) / entry_price
                    else:
                        self.floating_pnl = notional * (entry_price - price_close) / entry_price
                    current_total = self.waterfall.get_total_value() + self.floating_pnl
                else:
                    self.floating_pnl = 0.0
                    current_total = self.waterfall.get_total_value()
                
                # 更新抽水机制
                self.trading_capital = self.waterfall.update(current_total, ts)
                self.total_value = self.waterfall.get_total_value() + self.floating_pnl

            # === 2) 持仓管理 ===
            if side != "FLAT" and trade_type is not None:
                exit_reason = None
                exit_price = None

                # 止损 / 止盈（盘中）
                if side == "LONG":
                    if price_low <= stop_price:
                        exit_price = stop_price
                        exit_reason = "SL"
                    elif price_high >= take_price:
                        exit_price = take_price
                        exit_reason = "TP"
                elif side == "SHORT":
                    if price_high >= stop_price:
                        exit_price = stop_price
                        exit_reason = "SL"
                    elif price_low <= take_price:
                        exit_price = take_price
                        exit_reason = "TP"

                # 持仓时间限制
                bars_held += 1
                if exit_price is None and max_bars_hold > 0 and bars_held >= max_bars_hold:
                    exit_price = price_close
                    exit_reason = "TIME"

                # 趋势结束/冲突：提前离场
                if exit_price is None and (trend_strength_now <= 0 or trend_dir_now == 0):
                    exit_price = price_close
                    exit_reason = "TREND_REV"

                if exit_price is not None:
                    # 计算实际盈亏（包含手续费和滑点）
                    if side == "LONG":
                        gross_pnl = notional * (exit_price - entry_price) / entry_price
                    else:
                        gross_pnl = notional * (entry_price - exit_price) / entry_price

                    # 计算手续费（开仓和平仓各一次）
                    fee_entry = notional * self.cfg.fee_rate
                    fee_exit = notional * self.cfg.fee_rate
                    
                    # 计算滑点成本（入场和出场都有滑点）
                    if side == "LONG":
                        slippage_cost = notional * self.cfg.slippage * 2  # 入场和出场各一次
                    else:
                        slippage_cost = notional * self.cfg.slippage * 2
                    
                    # 净盈亏
                    pnl = gross_pnl - fee_entry - fee_exit - slippage_cost
                    
                    # 更新资金
                    self.trading_capital += pnl
                    self.total_value = self.waterfall.get_total_value()
                    
                    # 计算盈亏百分比（相对于初始资金）
                    pnl_pct = pnl / max(self.waterfall.initial_capital, 1e-9)

                    # 计算布林带位置
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

                    # 记录交易
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

                    # 重置状态
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
                    self.floating_pnl = 0.0

            # === 3) 空仓 → 入场 ===
            if side == "FLAT":
                sig_side, sig_type = self._entry_signal(ts, row, prev_row)
                if sig_side != "FLAT" and sig_type is not None:
                    if atr_1h > 0 and self.trading_capital > 0:
                        # 计算入场价格（包含滑点）
                        if sig_side == "LONG":
                            entry_price = price_close * (1.0 + self.cfg.slippage)
                        else:
                            entry_price = price_close * (1.0 - self.cfg.slippage)

                        # 计算仓位
                        stop_price, take_price, notional, max_bars_hold, sl_abs, rr_used = self._compute_sl_tp_notional(
                            sig_side,
                            trend_strength_now,
                            entry_price,
                            atr_1h,
                            self.trading_capital,  # 使用抽水后的交易资金
                        )

                        if notional > 0 and stop_price > 0 and take_price > 0 and max_bars_hold > 0:
                            side = sig_side
                            trade_type = sig_type
                            entry_time = ts
                            atr_used_for_trade = atr_1h
                            rr_used_for_trade = rr_used
                            trend_dir_for_trade = trend_dir_now
                            trend_strength_for_trade = trend_strength_now

                            # 扣除开仓手续费
                            fee_entry = notional * self.cfg.fee_rate
                            self.trading_capital -= fee_entry
                            self.total_value = self.waterfall.get_total_value() - fee_entry
                            bars_held = 0
                        else:
                            # 重置入场状态
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

            # 记录资金曲线
            equity_series.append((ts, self.total_value))
            prev_row = row

        # 生成资金曲线
        if equity_series:
            idx, vals = zip(*equity_series)
            idx = pd.DatetimeIndex(idx)
            if idx.tz is not None:
                idx = idx.tz_convert(None)
            self.equity_curve = pd.Series(vals, index=idx, name="equity")

        if plot:
            self._plot_equity()

    # ========= 结果分析 =========
    def summary(self):
        if self.equity_curve is None:
            raise ValueError("请先运行 run_backtest()")
        
        equity = self.equity_curve
        trades = self.trades
        
        if len(equity) == 0:
            print("没有产生任何交易。")
            return

        # 基本统计
        total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
        days = (equity.index[-1] - equity.index[0]).days
        if days <= 0:
            annual_return = np.nan
        else:
            annual_return = (1 + total_return) ** (365.0 / days) - 1.0

        # 回撤计算
        equity_cummax = equity.cummax()
        drawdown = equity / equity_cummax - 1.0
        max_dd = drawdown.min()

        # 交易统计
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / len(trades) if trades else 0.0
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0.0
        rr_real = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan

        print("=" * 60)
        print("V31 规则版 Teacher_V9 · 抽水机制版 回测结果")
        print("=" * 60)
        print(f"交易对: {self.cfg.symbol}")
        print(f"测试天数: {self.cfg.days}")
        print(f"杠杆倍数: {self.cfg.leverage}")
        print(f"初始资金: ${self.cfg.initial_equity:,.2f}")
        print(f"最终总资产: ${equity.iloc[-1]:,.2f}")
        print(f"总收益率: {total_return*100:.2f}%")
        print(f"年化收益: {annual_return*100:.2f}%")
        print(f"最大回撤: {max_dd*100:.2f}%")
        print(f"交易次数: {len(trades)}")
        print(f"胜率: {win_rate*100:.2f}%")
        print(f"平均盈利: ${avg_win:.2f}")
        print(f"平均亏损: ${avg_loss:.2f}")
        print(f"实际盈亏比: {rr_real:.2f}")
        print(f"手续费率: {self.cfg.fee_rate*100:.4f}%")
        print(f"滑点: {self.cfg.slippage*100:.4f}%")
        
        # 抽水机制统计
        if self.cfg.enable_waterfall:
            print("\n" + "-" * 40)
            print("抽水机制统计")
            print("-" * 40)
            print(f"交易资金: ${self.waterfall.trading_capital:,.2f}")
            print(f"利润池可提取: ${self.waterfall.profit_pool:,.2f}")
            print(f"抽水触发次数: {len(self.waterfall.history)}")
            if self.waterfall.history:
                total_withdrawn = sum(record['withdrawn'] for record in self.waterfall.history)
                avg_growth = sum(record['growth_pct'] for record in self.waterfall.history) / len(self.waterfall.history)
                print(f"总抽水金额: ${total_withdrawn:,.2f}")
                print(f"平均每次增长: {avg_growth*100:.2f}%")
            print(f"抽水比例: {self.waterfall.withdraw_rate*100}%")
            print(f"增长触发阈值: {self.waterfall.growth_threshold*100}%")
        
        print("=" * 60)

    def export_trades(self, filepath: str):
        if not self.trades:
            print("无交易可导出。")
            return
        
        rows = []
        for t in self.trades:
            rows.append({
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
            })
        
        df = pd.DataFrame(rows)
        
        # 处理时区
        for col in ["entry_time", "exit_time"]:
            if col in df.columns and np.issubdtype(df[col].dtype, np.datetime64):
                try:
                    df[col] = df[col].dt.tz_convert(None)
                except Exception:
                    pass
        
        # 导出
        if filepath.lower().endswith(".xlsx"):
            df.to_excel(filepath, index=False)
        else:
            df.to_csv(filepath, index=False)
        
        print(f"交易明细已导出到: {filepath}")

    def _plot_equity(self):
        if self.equity_curve is None or len(self.equity_curve) == 0:
            print("无资金曲线可绘制")
            return
        
        eq = self.equity_curve
        
        plt.figure(figsize=(14, 8))
        
        # 资金曲线
        plt.subplot(2, 1, 1)
        plt.plot(eq.index, eq.values, 'b-', linewidth=1.5)
        
        # 标记抽水事件
        if self.cfg.enable_waterfall and self.waterfall.history:
            for record in self.waterfall.history:
                plt.axvline(x=record['timestamp'], color='r', alpha=0.3, linestyle='--', linewidth=0.5)
        
        plt.title(f"V31 Teacher_V9 抽水机制版 · {self.cfg.symbol} (杠杆:{self.cfg.leverage}x)")
        plt.xlabel("时间")
        plt.ylabel("总资产 (USDT)")
        plt.grid(True, alpha=0.3)
        
        # 回撤曲线
        plt.subplot(2, 1, 2)
        equity_cummax = eq.cummax()
        drawdown = (eq / equity_cummax - 1.0) * 100
        plt.fill_between(eq.index, drawdown, 0, color='red', alpha=0.3)
        plt.plot(eq.index, drawdown, 'r-', linewidth=1)
        plt.title("回撤曲线")
        plt.xlabel("时间")
        plt.ylabel("回撤 (%)")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="V31 Teacher_V9 · 抽水机制版 回测")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="交易对")
    parser.add_argument("--days", type=int, default=365, help="回测天数")
    parser.add_argument("--leverage", type=float, default=3.0, help="杠杆倍数")  # 新增杠杆参数
    parser.add_argument("--no-plot", action="store_true", help="不显示图表")
    parser.add_argument("--no-waterfall", action="store_true", help="禁用抽水机制")
    parser.add_argument("--withdraw-rate", type=float, default=0.10, help="抽水比例")
    parser.add_argument("--growth-threshold", type=float, default=0.01, help="增长触发阈值")
    parser.add_argument("--export", type=str, default="", help="导出交易明细")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 创建配置
    cfg = V31V9Config(
        symbol=args.symbol.upper(), 
        days=args.days,
        leverage=args.leverage,  # 使用命令行传入的杠杆
        enable_waterfall=not args.no_waterfall,
        waterfall_withdraw_rate=args.withdraw_rate,
        waterfall_growth_threshold=args.growth_threshold
    )
    
    # 运行回测
    system = V31RuleTrendSystemV9Waterfall(cfg)
    system.run_backtest(plot=not args.no_plot)
    system.summary()
    
    # 导出结果
    if args.export:
        system.export_trades(args.export)