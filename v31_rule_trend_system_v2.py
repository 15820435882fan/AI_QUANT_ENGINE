# v31_rule_trend_system_v2.py
# V31 · 规则版 Teacher_V2 · Option1 增强版
#
# 特点：
#   - 以 5m 为主周期执行交易（更“丝滑”）
#   - 从 5m 重采样出 15m、4H 做趋势与大方向判断
#   - 4H EMA20/50 判多空方向；15m EMA20/50 辅助；
#   - 5m 上用 BOLL + MACD 做入场，条件有所放宽（更容易出信号）
#   - 固定风险 + 紧止损 + 名义仓位上限 = equity * leverage → 间接模拟杠杆
#   - 单向持仓，RR 固定，可回测、可解释，后续可用于生成 Teacher 标签
#
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from local_data_engine_v22_9 import LocalDataEngineV22_9


Side = Literal["FLAT", "LONG", "SHORT"]


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: Side
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

    # 大周期参数（4H）
    macro_ema_fast: int = 20
    macro_ema_slow: int = 50

    # 中周期参数（15m）
    trend_ema_fast: int = 20
    trend_ema_slow: int = 50

    # 入场周期参数（5m）
    boll_window: int = 20
    boll_k: float = 2.0
    boll_touch_buffer: float = 0.005   # 0.5% 容忍度，放宽“触及布林带”的条件
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # 风控与仓位
    risk_per_trade: float = 0.02       # 目标：单笔最多亏总权益的 2%
    rr: float = 2.0                    # RR = 2 → TP 距离是 SL 的 2 倍
    sl_pct: float = 0.003              # 初始止损距离：0.3%（更适合 5m）
    max_bars_hold: int = 96            # 最多持仓 96 根 5m（约 8 小时）
    leverage: float = 3.0              # 名义仓位上限：equity * leverage

    fee_rate: float = 0.0004           # 手续费 0.04%
    slippage: float = 0.0002           # 滑点 0.02%（简单模型）


class V31RuleTrendSystemV2:
    def __init__(self, cfg: V31Config):
        self.cfg = cfg
        self.df_5m: Optional[pd.DataFrame] = None
        self.trades: List[Trade] = []
        self.equity_curve: Optional[pd.Series] = None

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

        # 1）4H 多空方向（从 5m 重采样）
        ohlc_dict = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        df_4h = df5.resample("4H").agg(ohlc_dict).dropna()
        ema_fast_4h = df_4h["close"].ewm(span=self.cfg.macro_ema_fast, adjust=False).mean()
        ema_slow_4h = df_4h["close"].ewm(span=self.cfg.macro_ema_slow, adjust=False).mean()
        macro_dir_4h = np.sign(ema_fast_4h - ema_slow_4h)  # +1 多, -1 空, 0 中性

        macro_dir = macro_dir_4h.reindex(df5.index, method="ffill")
        macro_dir = macro_dir.fillna(method="bfill").fillna(0.0)
        df5["macro_dir"] = macro_dir

        # 2）15m 中周期趋势（EMA20/50），再映射到 5m
        df_15m = df5.resample("15T").agg(ohlc_dict).dropna()
        ema_fast_15 = df_15m["close"].ewm(span=self.cfg.trend_ema_fast, adjust=False).mean()
        ema_slow_15 = df_15m["close"].ewm(span=self.cfg.trend_ema_slow, adjust=False).mean()
        trend_dir_15 = np.sign(ema_fast_15 - ema_slow_15)  # +1 多, -1 空, 0 中性

        trend_dir = trend_dir_15.reindex(df5.index, method="ffill")
        trend_dir = trend_dir.fillna(method="bfill").fillna(0.0)
        df5["trend_dir"] = trend_dir

        # 3）5m BOLL
        mid = close5.rolling(self.cfg.boll_window).mean()
        std = close5.rolling(self.cfg.boll_window).std()
        upper = mid + self.cfg.boll_k * std
        lower = mid - self.cfg.boll_k * std
        df5["boll_mid"] = mid
        df5["boll_up"] = upper
        df5["boll_low"] = lower

        # 4）5m MACD
        ema_fast_5 = close5.ewm(span=self.cfg.macd_fast, adjust=False).mean()
        ema_slow_5 = close5.ewm(span=self.cfg.macd_slow, adjust=False).mean()
        macd = ema_fast_5 - ema_slow_5
        macd_signal = macd.ewm(span=self.cfg.macd_signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        df5["macd"] = macd
        df5["macd_signal"] = macd_signal
        df5["macd_hist"] = macd_hist

        self.df_5m = df5.dropna().copy()

    # -------- 入场信号 --------
    def _entry_signal(self, row: pd.Series, prev_row: pd.Series) -> Side:
        """
        基于：
          - macro_dir: 4H 大方向（EMA20/50）
          - trend_dir: 15m 趋势方向（EMA20/50）
          - BOLL: 价格接近上下轨（允许 0.5% 缓冲）
          - MACD: 金叉 / 死叉 或 histogram 同向
        返回：
          - "LONG" / "SHORT" / "FLAT"
        """
        macro_dir = row["macro_dir"]
        trend_dir = row["trend_dir"]
        close = row["close"]
        boll_up = row["boll_up"]
        boll_low = row["boll_low"]
        macd_hist = row["macd_hist"]
        macd_hist_prev = prev_row["macd_hist"]

        same_trend = (macro_dir * trend_dir) >= 0

        buf = self.cfg.boll_touch_buffer
        near_lower = close <= boll_low * (1.0 + buf)
        near_upper = close >= boll_up * (1.0 - buf)

        gold_cross = (macd_hist > 0) and (macd_hist_prev <= 0)
        dead_cross = (macd_hist < 0) and (macd_hist_prev >= 0)

        if macro_dir > 0 and same_trend and near_lower and (gold_cross or macd_hist > 0):
            return "LONG"

        if macro_dir < 0 and same_trend and near_upper and (dead_cross or macd_hist < 0):
            return "SHORT"

        return "FLAT"

    # -------- 回测主循环 --------
    def run_backtest(self, plot: bool = False):
        if self.df_5m is None:
            self.load_data()
        self._calc_indicators()
        df = self.df_5m.copy()

        equity = self.cfg.initial_equity
        equity_series = []

        side: Side = "FLAT"
        entry_price = 0.0
        notional = 0.0
        stop_price = 0.0
        take_price = 0.0
        bars_held = 0
        entry_time: Optional[pd.Timestamp] = None

        prev_row = df.iloc[0]

        for ts, row in df.iloc[1:].iterrows():
            price_high = row["high"]
            price_low = row["low"]
            price_close = row["close"]

            # 1）持仓管理：检查平仓条件
            if side != "FLAT":
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
                if exit_price is None and bars_held >= self.cfg.max_bars_hold:
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

                    self.trades.append(
                        Trade(
                            entry_time=entry_time,
                            exit_time=ts,
                            side=side,
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
                    entry_price = 0.0
                    notional = 0.0
                    stop_price = 0.0
                    take_price = 0.0
                    bars_held = 0
                    entry_time = None

            # 2）若空仓，则检查是否有新的入场信号
            if side == "FLAT":
                signal = self._entry_signal(row, prev_row)
                if signal in ("LONG", "SHORT"):
                    side = signal
                    entry_price = price_close
                    entry_time = ts

                    sl_pct = self.cfg.sl_pct
                    if side == "LONG":
                        stop_price = entry_price * (1.0 - sl_pct)
                        take_price = entry_price * (1.0 + sl_pct * self.cfg.rr)
                    else:
                        stop_price = entry_price * (1.0 + sl_pct)
                        take_price = entry_price * (1.0 - sl_pct * self.cfg.rr)

                    risk_target = equity * self.cfg.risk_per_trade
                    stop_dist = abs(entry_price - stop_price)
                    if stop_dist <= 0:
                        side = "FLAT"
                        entry_price = 0.0
                        entry_time = None
                    else:
                        loss_per_unit_notional = stop_dist / entry_price
                        notional_target = risk_target / loss_per_unit_notional
                        max_notional = equity * self.cfg.leverage
                        notional = min(notional_target, max_notional)

                        if notional <= 0:
                            side = "FLAT"
                            entry_price = 0.0
                            entry_time = None
                        else:
                            fee_entry = notional * self.cfg.fee_rate
                            equity -= fee_entry
                            bars_held = 0

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

        print("===== V31 规则版 Teacher_V2 · 5m 增强版 回测结果 =====")
        print(f"交易对: {self.cfg.symbol}, 天数: {self.cfg.days}")
        print(f"初始资金: {self.cfg.initial_equity:,.2f}")
        print(f"最终资金: {equity.iloc[-1]:,.2f}")
        print(f"总收益率: {total_return*100:.2f}%")
        print(f"年化收益: {annual_return*100:.2f}%")
        print(f"最大回撤: {max_dd*100:.2f}%")
        print(f"交易笔数: {len(trades)}")
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
        plt.title(f"V31 规则版 Teacher_V2 资金曲线 (5m, {self.cfg.symbol})")
        plt.xlabel("Time")
        plt.ylabel("Equity (USDT)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="V31 规则版 Teacher_V2 · 5m 增强版 回测")
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = V31Config(symbol=args.symbol.upper(), days=args.days)
    system = V31RuleTrendSystemV2(cfg)
    system.run_backtest(plot=not args.no_plot)
    system.summary()
