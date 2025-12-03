# v31_rule_trend_system_v7.py
# V31 · 规则版 Teacher_V7
# 纯趋势跟随版（无反转）· 高频趋势模式
#
# 核心设计：
# - 4H + 1H 决定方向 & 趋势强度（regime）
# - 仅做顺势单（TREND），不做任何“抄底/反转”
# - 5m 作为执行周期，高频顺势入场（右侧信号）
# - 15m 作为趋势确认滤波（可选）
# - 动态 RR：强趋势 RR≈4，中趋势≈3，弱趋势≈2.5
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
    macro_dir: float
    trend_dir: float
    regime_level: int  # 2=强趋势,1=中趋势,0=弱趋势,-1=冲突
    macd_hist_5m: float
    boll_pos_5m: float
    atr_used: float
    rr_used: float


@dataclass
class V31V7Config:
    symbol: str = "BTCUSDT"
    days: int = 365
    initial_equity: float = 10_000.0

    # 大周期（4H）
    macro_ema_fast: int = 20
    macro_ema_slow: int = 50

    # 中周期（1H）
    trend_ema_fast: int = 20
    trend_ema_slow: int = 50

    # 趋势强度阈值
    # strength = abs(ema_fast - ema_slow) / close
    strong_th: float = 0.02   # >=2% 强趋势
    medium_th: float = 0.01   # 1%~2% 中等趋势
    weak_th: float = 0.005    # 0.5%~1% 弱趋势，仍可顺势小仓参与

    # 执行周期（5m）
    exec_tf: str = "5m"
    boll_window_5m: int = 20
    boll_k_5m: float = 2.0
    boll_pullback_factor: float = 0.3  # 趋势中允许回调靠近下/上轨的比例 [0~1]

    macd_fast_5m: int = 12
    macd_slow_5m: int = 26
    macd_signal_5m: int = 9

    ema_fast_5m: int = 10
    ema_slow_5m: int = 30

    # 15m 过滤（趋势配合用）
    trend_tf: str = "15T"
    macd_fast_15m: int = 12
    macd_slow_15m: int = 26
    macd_signal_15m: int = 9

    # ATR 止损
    atr_window_5m: int = 14
    atr_window_15m: int = 14
    sl_min_pct: float = 0.003  # 最小止损 0.3%

    # 风控
    risk_per_trade: float = 0.015  # 单笔风险 1.5%
    leverage: float = 3.0

    # 仓位上限（占杠杆名义余额）
    pos_cap_strong: float = 1.0
    pos_cap_medium: float = 0.7
    pos_cap_weak: float = 0.4

    # 最大持仓时间（以 5m bar 计）
    max_bars_strong: int = 96   # ≈8 小时
    max_bars_medium: int = 72
    max_bars_weak: int = 48

    fee_rate: float = 0.0004
    slippage: float = 0.0002


class V31RuleTrendSystemV7:
    def __init__(self, cfg: V31V7Config):
        self.cfg = cfg
        self.df_5m: Optional[pd.DataFrame] = None
        self.trades: List[Trade] = []
        self.equity_curve: Optional[pd.Series] = None

    # -------- 工具函数 --------
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

    # -------- 数据加载与指标 --------
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

        # ---- 4H 趋势方向 & 强度 ----
        df_4h = df5.resample("4H").agg(ohlc).dropna()
        ema_fast_4h = df_4h["close"].ewm(span=self.cfg.macro_ema_fast, adjust=False).mean()
        ema_slow_4h = df_4h["close"].ewm(span=self.cfg.macro_ema_slow, adjust=False).mean()
        macro_dir_4h = np.sign(ema_fast_4h - ema_slow_4h)
        macro_strength_4h = (ema_fast_4h - ema_slow_4h).abs() / (df_4h["close"].abs() + 1e-12)

        # ---- 1H 趋势方向 & 强度 ----
        df_1h = df5.resample("1H").agg(ohlc).dropna()
        ema_fast_1h = df_1h["close"].ewm(span=self.cfg.trend_ema_fast, adjust=False).mean()
        ema_slow_1h = df_1h["close"].ewm(span=self.cfg.trend_ema_slow, adjust=False).mean()
        trend_dir_1h = np.sign(ema_fast_1h - ema_slow_1h)
        trend_strength_1h = (ema_fast_1h - ema_slow_1h).abs() / (df_1h["close"].abs() + 1e-12)

        # 对齐到 5m
        macro_dir = macro_dir_4h.reindex(df5.index, method="ffill")
        macro_dir = macro_dir.fillna(method="bfill").fillna(0.0)
        macro_str = macro_strength_4h.reindex(df5.index, method="ffill").fillna(0.0)

        trend_dir = trend_dir_1h.reindex(df5.index, method="ffill")
        trend_dir = trend_dir.fillna(method="bfill").fillna(0.0)
        trend_str = trend_strength_1h.reindex(df5.index, method="ffill").fillna(0.0)

        df5["macro_dir"] = macro_dir
        df5["macro_strength"] = macro_str
        df5["trend_dir"] = trend_dir
        df5["trend_strength"] = trend_str

        combined_str = np.minimum(macro_str, trend_str)
        df5["combined_strength"] = combined_str

        regime_level = np.zeros(len(df5), dtype=int)
        same_sign = (macro_dir * trend_dir) > 0
        conflict = (macro_dir * trend_dir) <= 0

        strong_mask = same_sign & (combined_str >= self.cfg.strong_th)
        medium_mask = same_sign & (combined_str >= self.cfg.medium_th) & (combined_str < self.cfg.strong_th)
        weak_mask = same_sign & (combined_str >= self.cfg.weak_th) & (combined_str < self.cfg.medium_th)

        regime_level[strong_mask] = 2
        regime_level[medium_mask] = 1
        regime_level[weak_mask] = 0
        regime_level[conflict] = -1

        df5["regime_level"] = regime_level

        # ---- 5m BOLL, MACD, EMA ----
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

        # ---- 15m MACD（趋势过滤） ----
        df_15m = df5.resample(self.cfg.trend_tf).agg(ohlc).dropna()
        close15 = df_15m["close"]

        ema_fast_15 = close15.ewm(span=self.cfg.macd_fast_15m, adjust=False).mean()
        ema_slow_15 = close15.ewm(span=self.cfg.macd_slow_15m, adjust=False).mean()
        macd15 = ema_fast_15 - ema_slow_15
        macd15_sig = macd15.ewm(span=self.cfg.macd_signal_15m, adjust=False).mean()
        macd15_hist = macd15 - macd15_sig

        df5["macd_hist_15m"] = macd15_hist.reindex(df5.index, method="ffill")

        # ---- ATR ----
        atr5 = self._atr(df5, self.cfg.atr_window_5m)
        atr15_raw = self._atr(df_15m, self.cfg.atr_window_15m)
        atr15 = atr15_raw.reindex(df5.index, method="ffill")

        df5["atr_5m"] = atr5
        df5["atr_15m"] = atr15

        # 标记 15m bar 起点（方便控制入场节奏）
        df5["is_15m_bar"] = (df5.index.minute % 15 == 0)

        self.df_5m = df5.dropna().copy()

    # -------- 入场信号（纯趋势） --------
    def _entry_signal(
        self,
        ts: pd.Timestamp,
        row: pd.Series,
        prev_row: pd.Series,
    ) -> Tuple[Side, Optional[TradeType]]:
        cfg = self.cfg

        regime_level = int(row["regime_level"])
        macro_dir = row["macro_dir"]
        trend_dir = row["trend_dir"]
        if regime_level < 0 or macro_dir == 0 or trend_dir == 0:
            return "FLAT", None

        close = row["close"]
        mid5 = row["boll_mid_5m"]
        up5 = row["boll_up_5m"]
        low5 = row["boll_low_5m"]
        ema_fast_5 = row["ema_fast_5m"]
        ema_slow_5 = row["ema_slow_5m"]
        macd_5 = row["macd_hist_5m"]
        macd_5_prev = prev_row["macd_hist_5m"]
        macd_15 = row["macd_hist_15m"]
        is_15m_bar = bool(row["is_15m_bar"])

        if mid5 <= 0 or up5 <= 0 or low5 <= 0:
            return "FLAT", None

        # 价格在 BOLL 区间的位置 [0=low, 1=up]
        rng = (up5 - low5)
        if rng == 0:
            return "FLAT", None
        boll_pos = (close - low5) / rng

        # 高频趋势模式：
        # - 只顺势（4H + 1H 同向）
        # - 15m MACD 与趋势方向同向
        # - 5m MACD 右侧增强 + 价格在趋势有利区间

        # -------- 多头趋势 --------
        if macro_dir > 0 and trend_dir > 0 and macd_15 > 0:
            # 条件1：价格在 BOLL 中上区间（不追太离谱）
            #   boll_pos ∈ [0.3, 0.9]
            cond_price = (boll_pos >= 0.3) and (boll_pos <= 0.9)
            # 条件2：5m MACD 右侧转强
            cond_macd = (macd_5 > 0) and (macd_5 >= macd_5_prev)
            # 条件3：5m EMA 快线在慢线上方
            cond_ema = ema_fast_5 > ema_slow_5

            # 入场节奏控制：只在 15m bar 起点才允许开新仓，避免过度频繁
            if cond_price and cond_macd and cond_ema and is_15m_bar:
                return "LONG", "TREND"

        # -------- 空头趋势 --------
        if macro_dir < 0 and trend_dir < 0 and macd_15 < 0:
            # 价格在 BOLL 中下区间
            cond_price = (boll_pos <= 0.7) and (boll_pos >= 0.1)
            cond_macd = (macd_5 < 0) and (macd_5 <= macd_5_prev)
            cond_ema = ema_fast_5 < ema_slow_5
            if cond_price and cond_macd and cond_ema and is_15m_bar:
                return "SHORT", "TREND"

        return "FLAT", None

    # -------- 仓位 & 止损 / RR --------
    def _compute_sl_tp_notional(
        self,
        side: Side,
        regime_level: int,
        entry_price: float,
        atr_value: float,
        equity: float,
        macro_dir: float,
    ) -> Tuple[float, float, float, int, float, float]:
        cfg = self.cfg
        if atr_value <= 0 or equity <= 0:
            return 0.0, 0.0, 0.0, 0, 0.0, 0.0

        # 动态 RR、止损倍数、仓位上限
        if regime_level >= 2:  # 强趋势
            rr = 4.0
            sl_mult = 3.0
            pos_cap = cfg.pos_cap_strong
            max_bars = cfg.max_bars_strong
        elif regime_level == 1:  # 中趋势
            rr = 3.0
            sl_mult = 2.5
            pos_cap = cfg.pos_cap_medium
            max_bars = cfg.max_bars_medium
        else:  # 弱趋势
            rr = 2.5
            sl_mult = 2.0
            pos_cap = cfg.pos_cap_weak
            max_bars = cfg.max_bars_weak

        sl_dist_abs = atr_value * sl_mult
        if entry_price <= 0:
            return 0.0, 0.0, 0.0, 0, 0.0, 0.0
        sl_dist_pct_from_atr = sl_dist_abs / entry_price
        sl_dist_pct = max(sl_dist_pct_from_atr, cfg.sl_min_pct)
        sl_dist_abs = sl_dist_pct * entry_price

        if side == "LONG":
            stop_price = entry_price - sl_dist_abs
            take_price = entry_price + sl_dist_abs * rr
        else:
            stop_price = entry_price + sl_dist_abs
            take_price = entry_price - sl_dist_abs * rr

        if stop_price <= 0 or take_price <= 0:
            return 0.0, 0.0, 0.0, 0, 0.0, 0.0

        # 风险控制：按风险预算反推名义仓位
        risk_target = equity * cfg.risk_per_trade
        stop_dist_pct = abs(entry_price - stop_price) / entry_price
        if stop_dist_pct <= 0:
            return 0.0, 0.0, 0.0, 0, 0.0, 0.0

        notional_target = risk_target / stop_dist_pct

        # 与 4H 同向 → 不减仓；逆向 → 减半
        if (macro_dir > 0 and side == "LONG") or (macro_dir < 0 and side == "SHORT"):
            dir_factor = 1.0
        else:
            dir_factor = 0.5

        max_notional = equity * cfg.leverage * pos_cap * dir_factor
        notional = min(notional_target, max_notional)

        if notional <= 0:
            return 0.0, 0.0, 0.0, 0, 0.0, 0.0

        return stop_price, take_price, notional, max_bars, sl_dist_abs, rr

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
        rr_used_for_trade: float = 0.0
        regime_level_for_trade: int = 0

        prev_row = df.iloc[0]

        for ts, row in df.iloc[1:].iterrows():
            price_high = row["high"]
            price_low = row["low"]
            price_close = row["close"]
            macro_dir = row["macro_dir"]
            regime_level_now = int(row["regime_level"])
            atr_5m = row["atr_5m"]
            atr_15m = row["atr_15m"]
            macd_5m = row["macd_hist_5m"]
            mid5 = row["boll_mid_5m"]
            up5 = row["boll_up_5m"]
            low5 = row["boll_low_5m"]

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

                # 趋势完全反转时，强制平仓（即使未触发 TP/SL/TIME）
                if exit_price is None and regime_level_now < 0:
                    exit_price = price_close
                    exit_reason = "TREND_REV"

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

                    rng = (up5 - low5) if (up5 - low5) != 0 else 1e-9
                    boll_pos = (price_close - low5) / rng

                    et = entry_time
                    xt = ts
                    try:
                        if et is not None and et.tzinfo is not None:
                            et = et.tz_localize(None)
                    except TypeError:
                        pass
                    try:
                        if xt is not None and xt.tzinfo is not None:
                            xt = xt.tz_localize(None)
                    except TypeError:
                        pass

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
                            macro_dir=row["macro_dir"],
                            trend_dir=row["trend_dir"],
                            regime_level=regime_level_for_trade,
                            macd_hist_5m=float(macd_5m),
                            boll_pos_5m=float(boll_pos),
                            atr_used=float(atr_used_for_trade),
                            rr_used=float(rr_used_for_trade),
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
                    rr_used_for_trade = 0.0
                    regime_level_for_trade = 0

            # 2）空仓 → 检查入场
            if side == "FLAT":
                sig_side, sig_type = self._entry_signal(ts, row, prev_row)
                if sig_side != "FLAT" and sig_type is not None:
                    # 趋势单：用 15m ATR 做主要波动衡量
                    atr_val = atr_15m if atr_15m > 0 else atr_5m
                    if atr_val > 0 and equity > 0:
                        if sig_side == "LONG":
                            entry_price = price_close * (1.0 + self.cfg.slippage)
                        else:
                            entry_price = price_close * (1.0 - self.cfg.slippage)

                        stop_price, take_price, notional, max_bars_hold, sl_abs, rr_used = self._compute_sl_tp_notional(
                            sig_side, regime_level_now, entry_price, atr_val, equity, macro_dir
                        )

                        if notional > 0 and stop_price > 0 and take_price > 0 and max_bars_hold > 0:
                            side = sig_side
                            trade_type = sig_type
                            entry_time = ts
                            atr_used_for_trade = atr_val
                            rr_used_for_trade = rr_used
                            regime_level_for_trade = regime_level_now

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
                            rr_used_for_trade = 0.0
                            regime_level_for_trade = 0

            equity_series.append((ts, equity))
            prev_row = row

        if equity_series:
            idx, vals = zip(*equity_series)
            idx = pd.DatetimeIndex(idx)
            try:
                if idx.tz is not None:
                    idx = idx.tz_localize(None)
            except TypeError:
                pass
            self.equity_curve = pd.Series(vals, index=idx, name="equity")
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

        print("===== V31 规则版 Teacher_V7 · 纯趋势跟随（高频） 回测结果 =====")
        print(f"交易对: {self.cfg.symbol}, 天数: {self.cfg.days}")
        print(f"初始资金: {self.cfg.initial_equity:,.2f}")
        print(f"最终资金: {equity.iloc[-1]:,.2f}")
        print(f"总收益率: {total_return*100:.2f}%")
        print(f"年化收益: {annual_return*100:.2f}%")
        print(f"最大回撤: {max_dd*100:.2f}%")
        print(f"交易总笔数: {len(trades)} (全部为趋势单 TREND)")
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
                    "regime_level": t.regime_level,
                    "macd_hist_5m": t.macd_hist_5m,
                    "boll_pos_5m": t.boll_pos_5m,
                    "atr_used": t.atr_used,
                    "rr_used": t.rr_used,
                }
            )
        df = pd.DataFrame(rows)

        for col in ["entry_time", "exit_time"]:
            if col in df.columns and np.issubdtype(df[col].dtype, np.datetime64):
                try:
                    df[col] = df[col].dt.tz_localize(None)
                except (TypeError, AttributeError):
                    try:
                        df[col] = df[col].dt.tz_convert(None)
                    except Exception:
                        pass

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
        plt.title(f"V31 Teacher_V7 资金曲线 · 纯趋势跟随（{self.cfg.symbol}, 5m）")
        plt.xlabel("Time")
        plt.ylabel("Equity (USDT)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="V31 Teacher_V7 · 纯趋势跟随（高频） 回测")
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--export", type=str, default="", help="导出交易明细（.xlsx 或 .csv）")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = V31V7Config(symbol=args.symbol.upper(), days=args.days)
    system = V31RuleTrendSystemV7(cfg)
    system.run_backtest(plot=not args.no_plot)
    system.summary()
    if args.export:
        system.export_trades(args.export)
