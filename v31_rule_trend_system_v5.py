# v31_rule_trend_system_v5.py
# V31 · 规则版 Teacher_V5
# 核心回到「趋势跟随」：4H/1H 大趋势 + 15m 趋势执行 + 5m 右侧反转仅在“非趋势区”(4H*1H<=0) 启用
#
# 用户选择：
#  1) 反转单：只在非趋势区触发（macro_dir * trend_dir <= 0）
#  2) 趋势单：保持 V4 条件不变（仍是 15m BOLL + MACD 顺势）
#  3) 止损参数：保持 V4，不调整
#
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from local_data_engine_v22_9 import LocalDataEngineV22_9


Side = Literal["FLAT", "LONG", "SHORT"]
TradeType = Literal["TREND", "REVERSAL"]


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: Side
    trade_type: TradeType
    entry_price: float
    exit_price: float
    notional: float  # 名义仓位金额（USDT）
    pnl: float
    pnl_pct: float
    bars_held: int
    reason: str
    macro_dir: float
    trend_dir: float
    stoch_rsi: float
    macd_hist: float
    boll_pos: float  # (close - mid) / (up - low + eps)
    atr_used: float


@dataclass
class V31Config:
    symbol: str = "BTCUSDT"
    days: int = 365
    initial_equity: float = 10_000.0

    # 大周期（4H）
    macro_ema_fast: int = 20
    macro_ema_slow: int = 50

    # 中周期（1H）
    trend_ema_fast: int = 20
    trend_ema_slow: int = 50

    # 趋势执行周期（15m）
    trend_tf: str = "15T"  # 15m
    trend_boll_window: int = 20
    trend_boll_k: float = 2.0
    trend_boll_buffer: float = 0.005  # 0.5% 容忍

    trend_macd_fast: int = 12
    trend_macd_slow: int = 26
    trend_macd_signal: int = 9

    # 反转执行周期（5m），指标仍在 5m 上计算
    rsi_period: int = 14
    stoch_period: int = 14
    stoch_smooth_k: int = 3
    stoch_low: float = 10.0
    stoch_high: float = 90.0
    boll_touch_buffer: float = 0.01  # 1%

    # ATR 止损
    atr_window_5m: int = 14
    atr_window_15m: int = 14
    sl_atr_mult_trend: float = 2.5
    sl_atr_mult_reversal: float = 2.5
    sl_min_pct: float = 0.003  # 保底 0.3%

    # RR
    rr_trend: float = 3.0       # 趋势单 RR
    rr_reversal: float = 1.5    # 反转单 RR

    # 风控与仓位
    risk_per_trade: float = 0.02    # 单笔风险 2%
    leverage: float = 3.0           # 杠杆上限（名义仓位上限）

    # 仓位系数
    trend_pos_cap: float = 1.0      # 趋势单最大仓位系数
    reversal_pos_cap: float = 0.2   # 反转单最大仓位系数

    max_bars_hold_trend: int = 96       # 约 8 小时（96 * 5m）
    max_bars_hold_reversal: int = 48    # 约 4 小时

    fee_rate: float = 0.0004       # 手续费 0.04%
    slippage: float = 0.0002       # 滑点 0.02%（简单模型）


class V31RuleTrendSystemV5:
    def __init__(self, cfg: V31Config):
        self.cfg = cfg
        self.df_5m: Optional[pd.DataFrame] = None
        self.trades: List[Trade] = []
        self.equity_curve: Optional[pd.Series] = None

    # -------- 工具函数 --------
    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _stoch_rsi(rsi: pd.Series, period: int, smooth_k: int) -> pd.Series:
        min_rsi = rsi.rolling(period).min()
        max_rsi = rsi.rolling(period).max()
        stoch = (rsi - min_rsi) / (max_rsi - min_rsi + 1e-12)
        k = stoch.rolling(smooth_k).mean()
        return k * 100.0

    @staticmethod
    def _atr(df: pd.DataFrame, window: int) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window).mean()
        return atr

    # -------- 数据加载与指标计算 --------
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

        # --- 大周期 4H：多空方向 ---
        ohlc = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        df_4h = df5.resample("4H").agg(ohlc).dropna()
        ema_fast_4h = df_4h["close"].ewm(span=self.cfg.macro_ema_fast, adjust=False).mean()
        ema_slow_4h = df_4h["close"].ewm(span=self.cfg.macro_ema_slow, adjust=False).mean()
        macro_dir_4h = np.sign(ema_fast_4h - ema_slow_4h)  # +1 多, -1 空, 0 中性

        macro_dir = macro_dir_4h.reindex(df5.index, method="ffill")
        macro_dir = macro_dir.fillna(method="bfill").fillna(0.0)
        df5["macro_dir"] = macro_dir

        # --- 中周期 1H：趋势方向 ---
        df_1h = df5.resample("1H").agg(ohlc).dropna()
        ema_fast_1h = df_1h["close"].ewm(span=self.cfg.trend_ema_fast, adjust=False).mean()
        ema_slow_1h = df_1h["close"].ewm(span=self.cfg.trend_ema_slow, adjust=False).mean()
        trend_dir_1h = np.sign(ema_fast_1h - ema_slow_1h)

        trend_dir = trend_dir_1h.reindex(df5.index, method="ffill")
        trend_dir = trend_dir.fillna(method="bfill").fillna(0.0)
        df5["trend_dir"] = trend_dir

        # 「非趋势区」定义：macro_dir * trend_dir <= 0
        conflict = (df5["macro_dir"] * df5["trend_dir"]) <= 0
        df5["conflict_zone"] = conflict.astype(int)  # 1=非趋势区，可做反转；0=趋势区，只做顺势

        # --- 趋势执行周期：15m BOLL + MACD ---
        df_15m = df5.resample(self.cfg.trend_tf).agg(ohlc).dropna()
        close15 = df_15m["close"]

        mid15 = close15.rolling(self.cfg.trend_boll_window).mean()
        std15 = close15.rolling(self.cfg.trend_boll_window).std()
        up15 = mid15 + self.cfg.trend_boll_k * std15
        low15 = mid15 - self.cfg.trend_boll_k * std15

        ema_fast_15 = close15.ewm(span=self.cfg.trend_macd_fast, adjust=False).mean()
        ema_slow_15 = close15.ewm(span=self.cfg.trend_macd_slow, adjust=False).mean()
        macd15 = ema_fast_15 - ema_slow_15
        macd_sig15 = macd15.ewm(span=self.cfg.trend_macd_signal, adjust=False).mean()
        macd_hist15 = macd15 - macd_sig15

        df5["trend_boll_mid"] = mid15.reindex(df5.index, method="ffill")
        df5["trend_boll_up"] = up15.reindex(df5.index, method="ffill")
        df5["trend_boll_low"] = low15.reindex(df5.index, method="ffill")
        df5["trend_macd_hist"] = macd_hist15.reindex(df5.index, method="ffill")

        # 标记 15m 边界（仅在这些时刻允许开趋势单）
        df5["is_15m_bar"] = (df5.index.minute % 15 == 0)

        # --- 5m BOLL（反转辅助） ---
        mid5 = close5.rolling(self.cfg.trend_boll_window).mean()
        std5 = close5.rolling(self.cfg.trend_boll_window).std()
        up5 = mid5 + self.cfg.trend_boll_k * std5
        low5 = mid5 - self.cfg.trend_boll_k * std5
        df5["boll_mid_5m"] = mid5
        df5["boll_up_5m"] = up5
        df5["boll_low_5m"] = low5

        # --- 5m MACD（反转辅助） ---
        ema_fast_5 = close5.ewm(span=self.cfg.trend_macd_fast, adjust=False).mean()
        ema_slow_5 = close5.ewm(span=self.cfg.trend_macd_slow, adjust=False).mean()
        macd5 = ema_fast_5 - ema_slow_5
        macd5_sig = macd5.ewm(span=self.cfg.trend_macd_signal, adjust=False).mean()
        macd5_hist = macd5 - macd5_sig
        df5["macd_hist_5m"] = macd5_hist

        # --- 5m Stoch RSI ---
        rsi5 = self._rsi(close5, self.cfg.rsi_period)
        stoch_rsi5 = self._stoch_rsi(rsi5, self.cfg.stoch_period, self.cfg.stoch_smooth_k)
        df5["stoch_rsi_5m"] = stoch_rsi5

        # --- ATR ---
        atr5 = self._atr(df5, self.cfg.atr_window_5m)
        atr15_raw = self._atr(df_15m, self.cfg.atr_window_15m)
        atr15 = atr15_raw.reindex(df5.index, method="ffill")

        df5["atr_5m"] = atr5
        df5["atr_15m"] = atr15

        self.df_5m = df5.dropna().copy()

    # -------- 入场信号：返回 side, trade_type --------
    def _entry_signal(
        self,
        ts: pd.Timestamp,
        row: pd.Series,
        prev_row: pd.Series,
    ) -> Tuple[Side, Optional[TradeType]]:
        cfg = self.cfg

        macro_dir = row["macro_dir"]
        trend_dir = row["trend_dir"]
        conflict_zone = bool(row["conflict_zone"])  # True = 非趋势区，可反转

        close = row["close"]
        # 15m BOLL & MACD（趋势）
        t_mid = row["trend_boll_mid"]
        t_up = row["trend_boll_up"]
        t_low = row["trend_boll_low"]
        t_macd_hist = row["trend_macd_hist"]

        # 5m BOLL & MACD & StochRSI（反转）
        mid5 = row["boll_mid_5m"]
        up5 = row["boll_up_5m"]
        low5 = row["boll_low_5m"]
        macd_hist_5m = row["macd_hist_5m"]
        macd_hist_5m_prev = prev_row["macd_hist_5m"]
        stoch = row["stoch_rsi_5m"]
        stoch_prev = prev_row["stoch_rsi_5m"]

        is_15m_bar = bool(row["is_15m_bar"])

        # ---------- 1）5m 反转信号（仅在「非趋势区」开放） ----------
        if conflict_zone:
            # 多头反转：右侧从极低处拐头 + 贴近 5m 下轨 + MACD 动能改善
            if (
                stoch_prev <= cfg.stoch_low
                and stoch > stoch_prev
                and close <= low5 * (1.0 + cfg.boll_touch_buffer)
                and macd_hist_5m > macd_hist_5m_prev
            ):
                return "LONG", "REVERSAL"

            # 空头反转：右侧从极高处拐头 + 贴近 5m 上轨 + MACD 动能转弱
            if (
                stoch_prev >= cfg.stoch_high
                and stoch < stoch_prev
                and close >= up5 * (1.0 - cfg.boll_touch_buffer)
                and macd_hist_5m < macd_hist_5m_prev
            ):
                return "SHORT", "REVERSAL"

        # ---------- 2）15m 趋势信号（仅在「趋势区」开放） ----------
        # 趋势区定义：macro_dir * trend_dir > 0
        if (macro_dir * trend_dir) > 0 and macro_dir != 0 and trend_dir != 0 and is_15m_bar:
            # 多头趋势：4H 多 + 1H 多 + 15m 在中上轨附近 + MACD 同向
            if macro_dir > 0 and trend_dir > 0:
                cond_price = t_mid * (1.0 - cfg.trend_boll_buffer) <= close <= t_up * (1.0 + cfg.trend_boll_buffer)
                cond_macd = t_macd_hist > 0
                if cond_price and cond_macd:
                    return "LONG", "TREND"

            # 空头趋势：4H 空 + 1H 空 + 15m 在中下轨附近 + MACD 同向
            if macro_dir < 0 and trend_dir < 0:
                cond_price = t_low * (1.0 - cfg.trend_boll_buffer) <= close <= t_mid * (1.0 + cfg.trend_boll_buffer)
                cond_macd = t_macd_hist < 0
                if cond_price and cond_macd:
                    return "SHORT", "TREND"

        return "FLAT", None

    # -------- 仓位与止损计算 --------
    def _compute_sl_tp_notional(
        self,
        side: Side,
        trade_type: TradeType,
        entry_price: float,
        atr_value: float,
        equity: float,
        macro_dir: float,
    ) -> Tuple[float, float, float, int, float]:
        cfg = self.cfg

        if atr_value <= 0 or equity <= 0:
            return 0.0, 0.0, 0.0, 0, 0.0

        if trade_type == "TREND":
            sl_mult = cfg.sl_atr_mult_trend
            rr = cfg.rr_trend
            max_bars = cfg.max_bars_hold_trend
            base_pos_cap = cfg.trend_pos_cap
        else:
            sl_mult = cfg.sl_atr_mult_reversal
            rr = cfg.rr_reversal
            max_bars = cfg.max_bars_hold_reversal
            base_pos_cap = cfg.reversal_pos_cap

        sl_dist_abs = atr_value * sl_mult
        sl_dist_pct_from_atr = sl_dist_abs / entry_price if entry_price > 0 else 0.0
        sl_dist_pct = max(sl_dist_pct_from_atr, cfg.sl_min_pct)
        sl_dist_abs = sl_dist_pct * entry_price

        if side == "LONG":
            stop_price = entry_price - sl_dist_abs
            take_price = entry_price + sl_dist_abs * rr
        else:
            stop_price = entry_price + sl_dist_abs
            take_price = entry_price - sl_dist_abs * rr

        if stop_price <= 0 or take_price <= 0:
            return 0.0, 0.0, 0.0, 0, 0.0

        risk_target = equity * cfg.risk_per_trade
        stop_dist_pct = abs(entry_price - stop_price) / entry_price
        if stop_dist_pct <= 0:
            return 0.0, 0.0, 0.0, 0, 0.0

        notional_target = risk_target / stop_dist_pct

        # 顺大趋势时给足仓位，逆大趋势减半
        if (macro_dir > 0 and side == "LONG") or (macro_dir < 0 and side == "SHORT"):
            pos_cap = base_pos_cap
        else:
            pos_cap = base_pos_cap * 0.5

        max_notional = equity * cfg.leverage * pos_cap
        notional = min(notional_target, max_notional)

        if notional <= 0:
            return 0.0, 0.0, 0.0, 0, 0.0

        return stop_price, take_price, notional, max_bars, sl_dist_abs

    # -------- 回测主循环 --------
    def run_backtest(self, plot: bool = False):
        if self.df_5m is None:
            self.load_data()
        self._calc_indicators()
        df = self.df_5m.copy()

        equity = self.cfg.initial_equity
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

        prev_row = df.iloc[0]

        for ts, row in df.iloc[1:].iterrows():
            price_high = row["high"]
            price_low = row["low"]
            price_close = row["close"]
            macro_dir = row["macro_dir"]

            atr_5m = row["atr_5m"]
            atr_15m = row["atr_15m"]

            # 1）持仓管理
            if side != "FLAT" and trade_type is not None:
                exit_reason = None
                exit_price = None

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

                bars_held += 1
                if exit_price is None and max_bars_hold > 0 and bars_held >= max_bars_hold:
                    exit_price = price_close
                    exit_reason = "TIME"

                if exit_price is not None:
                    if side == "LONG":
                        gross_pnl = notional * (exit_price - entry_price) / entry_price
                    else:
                        gross_pnl = notional * (entry_price - exit_price) / entry_price

                    fee_entry = notional * self.cfg.fee_rate
                    fee_exit = notional * self.cfg.fee_rate
                    pnl = gross_pnl - fee_entry - fee_exit

                    equity += pnl
                    pnl_pct = pnl / max(self.cfg.initial_equity, 1e-9)

                    stoch = row["stoch_rsi_5m"]
                    macd_hist = row["macd_hist_5m"]
                    mid5 = row["boll_mid_5m"]
                    up5 = row["boll_up_5m"]
                    low5 = row["boll_low_5m"]
                    rng = (up5 - low5) if (up5 - low5) != 0 else 1e-9
                    boll_pos = (price_close - mid5) / rng

                    self.trades.append(
                        Trade(
                            entry_time=entry_time,
                            exit_time=ts,
                            side=side,
                            trade_type=trade_type,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            notional=notional,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            bars_held=bars_held,
                            reason=exit_reason or "UNKNOWN",
                            macro_dir=macro_dir,
                            trend_dir=row["trend_dir"],
                            stoch_rsi=float(stoch),
                            macd_hist=float(macd_hist),
                            boll_pos=float(boll_pos),
                            atr_used=float(atr_used_for_trade),
                        )
                    )

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

            # 2）空仓时检查入场
            if side == "FLAT":
                sig_side, sig_type = self._entry_signal(ts, row, prev_row)
                if sig_side != "FLAT" and sig_type is not None:
                    atr_val = atr_15m if sig_type == "TREND" else atr_5m

                    if atr_val > 0 and equity > 0:
                        if sig_side == "LONG":
                            entry_price = price_close * (1.0 + self.cfg.slippage)
                        else:
                            entry_price = price_close * (1.0 - self.cfg.slippage)

                        stop_price, take_price, notional, max_bars_hold, sl_abs = self._compute_sl_tp_notional(
                            sig_side, sig_type, entry_price, atr_val, equity, macro_dir
                        )

                        if notional > 0 and stop_price > 0 and take_price > 0 and max_bars_hold > 0:
                            side = sig_side
                            trade_type = sig_type
                            entry_time = ts
                            atr_used_for_trade = atr_val

                            fee_entry = notional * self.cfg.fee_rate
                            equity -= fee_entry
                            bars_held = 0
                        else:
                            side = "FLAT"
                            trade_type = None
                            entry_price = 0.0
                            notional = 0.0
                            stop_price = 0.0
                            take_price = 0.0
                            max_bars_hold = 0
                            atr_used_for_trade = 0.0

            equity_series.append((ts, equity))
            prev_row = row

        if equity_series:
            idx, vals = zip(*equity_series)
            self.equity_curve = pd.Series(vals, index=pd.DatetimeIndex(idx), name="equity")
        else:
            self.equity_curve = pd.Series([], dtype=float)

        if plot:
            self._plot_equity()

    # -------- 结果分析 --------
    def summary(self):
        if self.equity_curve is None:
            raise ValueError("请先运行 run_backtest()")

        equity = self.equity_curve
        trades = self.trades
        if len(equity) == 0:
            print("没有产生任何交易。")
            return

        total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
        days = (equity.index[-1] - equity.index[0]).days
        if days <= 0:
            annual_return = np.nan
        else:
            annual_return = (1 + total_return) ** (365.0 / days) - 1.0

        equity_cummax = equity.cummax()
        drawdown = equity / equity_cummax - 1.0
        max_dd = drawdown.min()

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / len(trades) if trades else 0.0
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0.0
        rr_real = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan

        trend_trades = [t for t in trades if t.trade_type == "TREND"]
        rev_trades = [t for t in trades if t.trade_type == "REVERSAL"]

        print("===== V31 规则版 Teacher_V5 · 趋势15m + 5m右侧反转(仅非趋势区) 回测结果 =====")
        print(f"交易对: {self.cfg.symbol}, 天数: {self.cfg.days}")
        print(f"初始资金: {self.cfg.initial_equity:,.2f}")
        print(f"最终资金: {equity.iloc[-1]:,.2f}")
        print(f"总收益率: {total_return*100:.2f}%")
        print(f"年化收益: {annual_return*100:.2f}%")
        print(f"最大回撤: {max_dd*100:.2f}%")
        print(f"交易总笔数: {len(trades)} (趋势: {len(trend_trades)}, 反转: {len(rev_trades)})")
        print(f"胜率: {win_rate*100:.2f}%")
        print(f"平均盈利: {avg_win:.2f}")
        print(f"平均亏损: {avg_loss:.2f}")
        print(f"实际 RR: {rr_real:.2f}")
        print("========================================")

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
                    "macro_dir": t.macro_dir,
                    "trend_dir": t.trend_dir,
                    "stoch_rsi_5m": t.stoch_rsi,
                    "macd_hist_5m": t.macd_hist,
                    "boll_pos": t.boll_pos,
                    "atr_used": t.atr_used,
                }
            )
        df = pd.DataFrame(rows)
        if filepath.lower().endswith(".xlsx"):
            df.to_excel(filepath, index=False)
        else:
            df.to_csv(filepath, index=False)
        print(f"交易明细已导出到: {filepath}")

    def _plot_equity(self):
        assert self.equity_curve is not None
        eq = self.equity_curve

        plt.figure(figsize=(12, 5))
        plt.plot(eq.index, eq.values)
        plt.title(f"V31 规则版 Teacher_V5 资金曲线 (5m执行, {self.cfg.symbol})")
        plt.xlabel("Time")
        plt.ylabel("Equity (USDT)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="V31 规则版 Teacher_V5 · 趋势15m + 5m右侧反转(仅非趋势区) 回测")
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--export", type=str, default="", help="导出交易明细到指定文件（.xlsx 或 .csv）")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = V31Config(symbol=args.symbol.upper(), days=args.days)
    system = V31RuleTrendSystemV5(cfg)
    system.run_backtest(plot=not args.no_plot)
    system.summary()
    if args.export:
        system.export_trades(args.export)
