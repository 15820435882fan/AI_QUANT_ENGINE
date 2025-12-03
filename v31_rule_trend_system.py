# v31_rule_trend_system.py
# V31 · 规则版 Teacher_V2
#
# 思路：
#   - 用 15m K 线作为主驱动
#   - 从 15m 重采样出 4H 和 1D，做大周期方向判断（EMA20/EMA50）
#   - 小周期用 15m BOLL + MACD 做入场信号
#   - 单向持仓（同一时间只有一笔多/空），固定 RR（止盈止损），简单仓位管理
#   - 不依赖任何 AI 模型，是一个“可解释、可回测”的规则策略
#
# 依赖：
#   - pandas, numpy, matplotlib
#   - local_data_engine_v22_9.LocalDataEngineV22_9（优先从 feather 加载）
#
# 用法（在项目根目录）：
#   python v31_rule_trend_system.py --symbol BTCUSDT --days 730 --plot
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
    size: float  # 名义仓位金额（以 USDT 计）
    pnl: float
    pnl_pct: float
    bars_held: int


@dataclass
class V31Config:
    symbol: str = "BTCUSDT"
    days: int = 730
    initial_equity: float = 10_000.0

    # 大周期参数（从 15m 重采样出 4H / 1D）
    macro_tf_4h_ema_fast: int = 20
    macro_tf_4h_ema_slow: int = 50

    # 小周期参数（15m）
    boll_window: int = 20
    boll_k: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # 风控与仓位
    risk_per_trade: float = 0.02      # 单笔风险占总权益比例
    rr: float = 2.0                   # RR = 2 → 止盈距离是止损的 2 倍
    max_bars_hold: int = 96           # 最多持仓 96 根 15m（约 1 天）

    fee_rate: float = 0.0004          # 手续费 0.04%
    slippage: float = 0.0002          # 滑点 0.02%


class V31RuleTrendSystem:
    def __init__(self, cfg: V31Config):
        self.cfg = cfg
        self.df_15m: Optional[pd.DataFrame] = None
        self.trades: List[Trade] = []
        self.equity_curve: Optional[pd.Series] = None

    # -------- 数据加载与指标 --------
    def load_data(self):
        engine = LocalDataEngineV22_9()
        df = engine.load_klines(self.cfg.symbol, "15m", days=self.cfg.days)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("K线索引必须是 DatetimeIndex")
        df = df.sort_index()
        self.df_15m = df

    def _calc_indicators(self):
        assert self.df_15m is not None
        df = self.df_15m.copy()

        close = df["close"]

        # 1）从 15m 重采样出 4H，计算 EMA 方向
        ohlc_dict = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        df_4h = df.resample("4H").agg(ohlc_dict).dropna()

        ema_fast = df_4h["close"].ewm(span=self.cfg.macro_tf_4h_ema_fast, adjust=False).mean()
        ema_slow = df_4h["close"].ewm(span=self.cfg.macro_tf_4h_ema_slow, adjust=False).mean()
        macro_dir_4h = np.sign(ema_fast - ema_slow)  # +1 多, -1 空, 0 中性

        # 将 4H 方向映射回 15m 时间轴（前向填充）
        macro_dir = macro_dir_4h.reindex(df.index, method="ffill").fillna(0.0)
        df["macro_dir"] = macro_dir

        # 2）15m BOLL
        mid = close.rolling(self.cfg.boll_window).mean()
        std = close.rolling(self.cfg.boll_window).std()
        upper = mid + self.cfg.boll_k * std
        lower = mid - self.cfg.boll_k * std
        df["boll_mid"] = mid
        df["boll_up"] = upper
        df["boll_low"] = lower

        # 3）15m MACD
        ema_fast_15 = close.ewm(span=self.cfg.macd_fast, adjust=False).mean()
        ema_slow_15 = close.ewm(span=self.cfg.macd_slow, adjust=False).mean()
        macd = ema_fast_15 - ema_slow_15
        macd_signal = macd.ewm(span=self.cfg.macd_signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        df["macd"] = macd
        df["macd_signal"] = macd_signal
        df["macd_hist"] = macd_hist

        self.df_15m = df.dropna().copy()

    # -------- 交易规则 --------
    def _entry_signal(self, row: pd.Series, prev_row: pd.Series) -> Side:
        """
        基于：
          - macro_dir: 大周期方向（4H EMA20/50）
          - BOLL: 价格靠近上下轨
          - MACD: 金叉/死叉
        返回：
          - "LONG" / "SHORT" / "FLAT"（FLAT 表示无入场信号）
        """
        macro_dir = row["macro_dir"]
        close = row["close"]
        boll_up = row["boll_up"]
        boll_low = row["boll_low"]
        macd_hist = row["macd_hist"]
        macd_hist_prev = prev_row["macd_hist"]

        # MACD 金叉 / 死叉
        gold_cross = (macd_hist > 0) and (macd_hist_prev <= 0)
        dead_cross = (macd_hist < 0) and (macd_hist_prev >= 0)

        # 价格靠近布林带
        near_lower = close <= boll_low
        near_upper = close >= boll_up

        # 多头信号：大周期看多 + 价格触及下轨 + MACD 金叉
        if macro_dir > 0 and near_lower and gold_cross:
            return "LONG"

        # 空头信号：大周期看空 + 价格触及上轨 + MACD 死叉
        if macro_dir < 0 and near_upper and dead_cross:
            return "SHORT"

        return "FLAT"

    # -------- 回测主循环 --------
    def run_backtest(self, plot: bool = False):
        if self.df_15m is None:
            self.load_data()
        self._calc_indicators()
        df = self.df_15m.copy()

        equity = self.cfg.initial_equity
        equity_series = []

        side: Side = "FLAT"
        entry_price = 0.0
        size = 0.0
        stop_price = 0.0
        take_price = 0.0
        bars_held = 0
        entry_time: Optional[pd.Timestamp] = None

        prev_row = df.iloc[0]

        for ts, row in df.iloc[1:].iterrows():
            price_open = row["open"]
            price_high = row["high"]
            price_low = row["low"]
            price_close = row["close"]

            # 先检查平仓条件（如果有持仓）
            if side != "FLAT":
                exit_reason = None
                exit_price = None

                if side == "LONG":
                    # 止损
                    if price_low <= stop_price:
                        exit_price = stop_price
                        exit_reason = "SL"
                    # 止盈
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

                # 时间止损
                bars_held += 1
                if exit_price is None and bars_held >= self.cfg.max_bars_hold:
                    exit_price = price_close
                    exit_reason = "TIME"

                if exit_price is not None:
                    # 计算 PnL（多头为 +, 空头为 -）
                    if side == "LONG":
                        gross_pnl = (exit_price - entry_price) * (size / entry_price)
                    else:
                        gross_pnl = (entry_price - exit_price) * (size / entry_price)

                    fee_entry = size * self.cfg.fee_rate
                    fee_exit = size * self.cfg.fee_rate
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
                            size=size,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            bars_held=bars_held,
                        )
                    )

                    # 重置持仓状态
                    side = "FLAT"
                    entry_price = 0.0
                    size = 0.0
                    stop_price = 0.0
                    take_price = 0.0
                    bars_held = 0
                    entry_time = None

            # 然后如果当前是空仓，检查是否有新的入场信号
            if side == "FLAT":
                signal = self._entry_signal(row, prev_row)
                if signal in ("LONG", "SHORT"):
                    # 以当前 close 开仓
                    side = signal
                    entry_price = price_close
                    entry_time = ts

                    risk_amount = equity * self.cfg.risk_per_trade
                    # 先给一个简单的 price-based 止损：1% 距离
                    sl_pct = 0.01
                    if side == "LONG":
                        stop_price = entry_price * (1.0 - sl_pct)
                        take_price = entry_price * (1.0 + sl_pct * self.cfg.rr)
                    else:
                        stop_price = entry_price * (1.0 + sl_pct)
                        take_price = entry_price * (1.0 - sl_pct * self.cfg.rr)

                    # 仓位大小：以风险金额 / 止损距离
                    if side == "LONG":
                        per_unit_loss = entry_price - stop_price
                    else:
                        per_unit_loss = stop_price - entry_price
                    if per_unit_loss <= 0:
                        # 避免异常情况
                        size = 0.0
                        side = "FLAT"
                        entry_time = None
                    else:
                        units = risk_amount / per_unit_loss
                        size = units * entry_price  # 名义金额
                        # 入场扣一次手续费
                        fee_entry = size * self.cfg.fee_rate
                        equity -= fee_entry

                        bars_held = 0

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
        # 近似年化（按 365 天）
        days = (equity.index[-1] - equity.index[0]).days
        if days <= 0:
            annual_return = np.nan
        else:
            annual_return = (1 + total_return) ** (365.0 / days) - 1.0

        # 最大回撤
        equity_cummax = equity.cummax()
        drawdown = equity / equity_cummax - 1.0
        max_dd = drawdown.min()

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / len(trades) if trades else 0.0
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0.0
        rr_real = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan

        print("===== V31 规则版 Teacher_V2 回测结果 =====")
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
        plt.title(f"V31 规则版 Teacher_V2 资金曲线 ({self.cfg.symbol}, 15m)")
        plt.xlabel("Time")
        plt.ylabel("Equity (USDT)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="V31 规则版 Teacher_V2 回测")
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = V31Config(symbol=args.symbol.upper(), days=args.days)
    system = V31RuleTrendSystem(cfg)
    system.run_backtest(plot=not args.no_plot)
    system.summary()
