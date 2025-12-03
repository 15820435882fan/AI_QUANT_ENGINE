# v31_rule_trend_system_v3.py
# V31 · 规则版 Teacher_V3
# 趋势 + 反转 双系统融合
#
# 配置（结合用户选择）：
#   - 中周期：1H（A）
#   - 反转单仓位上限：20%（A）
#   - 趋势单 RR：3.0（C）
#   - 反转单 RR：1.5（A）
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

    # 入场周期（5m）
    boll_window: int = 20
    boll_k: float = 2.0
    boll_touch_buffer: float = 0.01   # 1% 容忍度，放宽“接近布林带”

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Stoch RSI 参数（用户选择默认：14/14/3）
    rsi_period: int = 14
    stoch_period: int = 14
    stoch_smooth_k: int = 3

    # ATR 止损
    atr_window: int = 14
    sl_atr_mult_trend: float = 1.5
    sl_atr_mult_reversal: float = 1.0

    # RR（用户选择）
    rr_trend: float = 3.0       # 趋势单 RR
    rr_reversal: float = 1.5    # 反转单 RR

    # 风控与仓位
    risk_per_trade: float = 0.02    # 单笔风险 2%
    leverage: float = 3.0           # 杠杆上限（名义仓位上限）

    # 仓位系数
    trend_pos_cap: float = 1.0      # 趋势单最大仓位系数
    reversal_pos_cap: float = 0.2   # 反转单最大仓位系数（用户选 A=0.2）

    max_bars_hold_trend: int = 96       # 趋势单最多持仓 96 根 5m（约 8 小时）
    max_bars_hold_reversal: int = 48    # 反转单最多持仓 48 根 5m（约 4 小时）

    fee_rate: float = 0.0004       # 手续费 0.04%
    slippage: float = 0.0002       # 滑点 0.02%（简单模型）


class V31RuleTrendSystemV3:
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

        # --- 中周期 1H：趋势方向 & 冲突区 ---
        df_1h = df5.resample("1H").agg(ohlc).dropna()
        ema_fast_1h = df_1h["close"].ewm(span=self.cfg.trend_ema_fast, adjust=False).mean()
        ema_slow_1h = df_1h["close"].ewm(span=self.cfg.trend_ema_slow, adjust=False).mean()
        trend_dir_1h = np.sign(ema_fast_1h - ema_slow_1h)

        trend_dir = trend_dir_1h.reindex(df5.index, method="ffill")
        trend_dir = trend_dir.fillna(method="bfill").fillna(0.0)
        df5["trend_dir"] = trend_dir

        conflict = (df5["macro_dir"] * df5["trend_dir"]) < 0
        df5["conflict_zone"] = conflict.astype(int)  # 1=冲突区

        # --- 5m BOLL ---
        mid = close5.rolling(self.cfg.boll_window).mean()
        std = close5.rolling(self.cfg.boll_window).std()
        upper = mid + self.cfg.boll_k * std
        lower = mid - self.cfg.boll_k * std
        df5["boll_mid"] = mid
        df5["boll_up"] = upper
        df5["boll_low"] = lower

        # --- 5m MACD ---
        ema_fast_5 = close5.ewm(span=self.cfg.macd_fast, adjust=False).mean()
        ema_slow_5 = close5.ewm(span=self.cfg.macd_slow, adjust=False).mean()
        macd = ema_fast_5 - ema_slow_5
        macd_signal = macd.ewm(span=self.cfg.macd_signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        df5["macd"] = macd
        df5["macd_signal"] = macd_signal
        df5["macd_hist"] = macd_hist

        # --- 5m Stoch RSI ---
        rsi = self._rsi(close5, self.cfg.rsi_period)
        stoch_rsi = self._stoch_rsi(rsi, self.cfg.stoch_period, self.cfg.stoch_smooth_k)
        df5["stoch_rsi"] = stoch_rsi

        # --- 5m ATR ---
        atr = self._atr(df5, self.cfg.atr_window)
        df5["atr"] = atr

        self.df_5m = df5.dropna().copy()

    # -------- 入场信号：返回 side, trade_type --------
    def _entry_signal(self, row: pd.Series, prev_row: pd.Series) -> Tuple[Side, Optional[TradeType]]:
        cfg = self.cfg

        macro_dir = row["macro_dir"]
        trend_dir = row["trend_dir"]
        conflict_zone = bool(row["conflict_zone"])

        close = row["close"]
        boll_mid = row["boll_mid"]
        boll_up = row["boll_up"]
        boll_low = row["boll_low"]

        macd_hist = row["macd_hist"]
        macd_hist_prev = prev_row["macd_hist"]

        stoch_rsi = row["stoch_rsi"]

        # --- 1) 反转信号（优先级高，永远小仓） ---
        # 多头反转：超卖区（<10），接近下轨 或 MACD 由负转正
        if stoch_rsi <= 10:
            cond_price = close <= boll_low * (1.0 + cfg.boll_touch_buffer)
            cond_macd = (macd_hist > 0) and (macd_hist_prev <= 0)
            if cond_price or cond_macd:
                return "LONG", "REVERSAL"

        # 空头反转：超买区（>90），接近上轨 或 MACD 由正转负
        if stoch_rsi >= 90:
            cond_price = close >= boll_up * (1.0 - cfg.boll_touch_buffer)
            cond_macd = (macd_hist < 0) and (macd_hist_prev >= 0)
            if cond_price or cond_macd:
                return "SHORT", "REVERSAL"

        # --- 2) 趋势跟随信号（仅在非冲突区） ---
        if not conflict_zone and macro_dir != 0 and trend_dir != 0:
            # 多头趋势：4H 多 + 1H 多 + 价格在中轨上方附近 + MACD 同向
            if macro_dir > 0 and trend_dir > 0:
                cond_price = boll_mid * 0.99 <= close <= boll_up * 1.01
                cond_macd = macd_hist > 0
                if cond_price and cond_macd:
                    return "LONG", "TREND"

            # 空头趋势：4H 空 + 1H 空 + 价格在中轨下方附近 + MACD 同向
            if macro_dir < 0 and trend_dir < 0:
                cond_price = boll_low * 0.99 <= close <= boll_mid * 1.01
                cond_macd = macd_hist < 0
                if cond_price and cond_macd:
                    return "SHORT", "TREND"

        # 冲突区 + 没有极端 StochRSI 的情况下，不开新趋势仓
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
    ) -> Tuple[float, float, float, int]:
        cfg = self.cfg

        if atr_value <= 0 or equity <= 0:
            return 0.0, 0.0, 0.0, 0

        # 不同类型单的参数
        if trade_type == "TREND":
            sl_mult = cfg.sl_atr_mult_trend
            rr = cfg.rr_trend
            max_bars = cfg.max_bars_hold_trend
            base_pos_cap = cfg.trend_pos_cap
        else:  # REVERSAL
            sl_mult = cfg.sl_atr_mult_reversal
            rr = cfg.rr_reversal
            max_bars = cfg.max_bars_hold_reversal
            base_pos_cap = cfg.reversal_pos_cap

        sl_dist = atr_value * sl_mult
        if sl_dist <= 0:
            return 0.0, 0.0, 0.0, 0

        if side == "LONG":
            stop_price = entry_price - sl_dist
            take_price = entry_price + sl_dist * rr
        else:
            stop_price = entry_price + sl_dist
            take_price = entry_price - sl_dist * rr

        if stop_price <= 0 or take_price <= 0:
            return 0.0, 0.0, 0.0, 0

        # 计算风险对应的名义仓位
        risk_target = equity * cfg.risk_per_trade
        stop_dist_pct = abs(entry_price - stop_price) / entry_price
        if stop_dist_pct <= 0:
            return 0.0, 0.0, 0.0, 0

        notional_target = risk_target / stop_dist_pct

        # 仓位上限控制（结合大周期方向）
        # macro_dir 与持仓方向一致 → 使用 base_pos_cap
        # macro_dir 与持仓方向相反 → 再打 0.5 折
        if (macro_dir > 0 and side == "LONG") or (macro_dir < 0 and side == "SHORT"):
            pos_cap = base_pos_cap
        else:
            pos_cap = base_pos_cap * 0.5

        max_notional = equity * cfg.leverage * pos_cap
        notional = min(notional_target, max_notional)

        if notional <= 0:
            return 0.0, 0.0, 0.0, 0

        return stop_price, take_price, notional, max_bars

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

        prev_row = df.iloc[0]

        for ts, row in df.iloc[1:].iterrows():
            price_high = row["high"]
            price_low = row["low"]
            price_close = row["close"]
            macro_dir = row["macro_dir"]
            atr_val = row["atr"]

            # 1）持仓管理：检查平仓条件
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
                    # 计算 PnL
                    if side == "LONG":
                        gross_pnl = notional * (exit_price - entry_price) / entry_price
                    else:
                        gross_pnl = notional * (entry_price - exit_price) / entry_price

                    fee_entry = notional * self.cfg.fee_rate
                    fee_exit = notional * self.cfg.fee_rate
                    pnl = gross_pnl - fee_entry - fee_exit

                    equity += pnl
                    pnl_pct = pnl / max(self.cfg.initial_equity, 1e-9)

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

            # 2）若空仓，则检查是否有新的入场信号
            if side == "FLAT":
                sig_side, sig_type = self._entry_signal(row, prev_row)
                if sig_side != "FLAT" and sig_type is not None and atr_val > 0:
                    # 考虑滑点调整入场价
                    if sig_side == "LONG":
                        entry_price = price_close * (1.0 + self.cfg.slippage)
                    else:
                        entry_price = price_close * (1.0 - self.cfg.slippage)

                    stop_price, take_price, notional, max_bars_hold = self._compute_sl_tp_notional(
                        sig_side, sig_type, entry_price, atr_val, equity, macro_dir
                    )

                    if notional > 0 and stop_price > 0 and take_price > 0 and max_bars_hold > 0:
                        side = sig_side
                        trade_type = sig_type
                        entry_time = ts

                        # 入场手续费
                        fee_entry = notional * self.cfg.fee_rate
                        equity -= fee_entry
                        bars_held = 0
                    else:
                        # 信号无效，忽略
                        side = "FLAT"
                        trade_type = None
                        entry_price = 0.0
                        notional = 0.0
                        stop_price = 0.0
                        take_price = 0.0
                        max_bars_hold = 0

            equity_series.append((ts, equity))
            prev_row = row

        # 构建权益曲线
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

        print("===== V31 规则版 Teacher_V3 · 趋势 + 反转 融合 回测结果 =====")
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

    def _plot_equity(self):
        assert self.equity_curve is not None
        eq = self.equity_curve

        plt.figure(figsize=(12, 5))
        plt.plot(eq.index, eq.values)
        plt.title(f"V31 规则版 Teacher_V3 资金曲线 (5m, {self.cfg.symbol})")
        plt.xlabel("Time")
        plt.ylabel("Equity (USDT)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="V31 规则版 Teacher_V3 · 趋势 + 反转 融合 回测")
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = V31Config(symbol=args.symbol.upper(), days=args.days)
    system = V31RuleTrendSystemV3(cfg)
    system.run_backtest(plot=not args.no_plot)
    system.summary()
