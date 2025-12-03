# v31_rule_trend_system_v9_waterfall_fixed_v2.py
# V31 · 规则版 Trend System · Teacher_V9
# 修正版（手续费&滑点统一在平仓结算 + 严格复利统计）

import numpy as np
import pandas as pd
import talib
from dataclasses import dataclass
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


# ========= 配置区域 =========

@dataclass
class Config:
    symbol: str = "BTCUSDT"
    initial_equity: float = 10_000.0
    risk_per_trade: float = 0.01
    leverage: float = 1.0
    fee_rate: float = 0.0004
    slippage: float = 0.0005
    max_consecutive_losses: int = 5
    max_daily_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.10
    enable_waterfall: bool = True
    waterfall_profit_share: float = 0.10
    waterfall_growth_threshold: float = 0.01
    compound_mode: bool = True

    # 数据路径
    data_5m_path: str = "./data/binance_futures_BTCUSDT_5m.parquet"
    data_1h_path: str = "./data/binance_futures_BTCUSDT_1h.parquet"

    # 指标参数
    hma_fast_period_1h: int = 25
    hma_slow_period_1h: int = 50
    hma_fast_period_4h: int = 25
    hma_slow_period_4h: int = 50
    adx_period_1h: int = 14
    boll_period_5m: int = 100
    boll_std_5m: float = 2.0

    # 时间过滤
    trading_start_hour: int = 0
    trading_end_hour: int = 24

    # 持仓时间
    max_bars_normal: int = 72
    max_bars_strong: int = 96

    # 仓位上限
    position_cap: float = 0.95


# ========= 工具函数 & 枚举 =========

class Side:
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class TradeType:
    TREND = "TREND"


@dataclass
class TradeRecord:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    trade_type: str
    entry_price: float
    exit_price: float
    notional: float
    pnl: float
    rr: float
    bars_held: int
    trend_strength: int
    trend_dir: int
    atr_1h: float
    exit_reason: str


# ========= 抽水机制（Profit Waterfall） =========

class ProfitWaterfall:
    """
    管理交易资金与利润池的抽水机制：
    - 交易资金：用于滚动复利的“战斗资金”
    - 利润池：阶段性锁定的利润，防止回吐
    """

    def __init__(
        self,
        initial_capital: float,
        profit_share: float = 0.10,
        growth_threshold: float = 0.01,
    ):
        """
        :param initial_capital: 初始资金
        :param profit_share: 每次抽水的比例（相对于抽水部分）
        :param growth_threshold: 交易资金相对于上次抽水基准的增长阈值
        """
        self.initial_capital = initial_capital
        self.trading_capital = initial_capital
        self.profit_pool = 0.0
        self.last_peak_trading_capital = initial_capital
        self.profit_share = profit_share
        self.growth_threshold = growth_threshold

        # 统计
        self.total_water_drawn = 0.0
        self.water_draw_times = 0

    def update_after_trade(self, new_trading_capital: float) -> float:
        """
        检查是否需要抽水，返回更新后的交易资金
        修正：只对已实现盈利的部分抽水，不影响回撤统计
        """
        self.trading_capital = new_trading_capital

        # 只有当交易资金超过上次抽水基准 * (1 + growth_threshold) 才触发抽水
        target_level = self.last_peak_trading_capital * (1.0 + self.growth_threshold)
        if self.trading_capital > target_level:
            # 超出部分
            excess = self.trading_capital - self.last_peak_trading_capital
            # 其中抽出 profit_share 比例作为利润池
            water = excess * self.profit_share
            self.trading_capital -= water
            self.profit_pool += water
            self.total_water_drawn += water
            self.water_draw_times += 1

            # 更新抽水基准
            self.last_peak_trading_capital = self.trading_capital

        return self.trading_capital

    def get_total_value(self) -> float:
        """当前总资产 = 交易资金 + 利润池"""
        return self.trading_capital + self.profit_pool


# ========= 主策略类 =========

class V31TrendSystemTeacherV9:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        # 数据
        self.df_5m: Optional[pd.DataFrame] = None
        self.df_1h: Optional[pd.DataFrame] = None
        self.df_4h: Optional[pd.DataFrame] = None

        # 交易状态
        self.trades: List[TradeRecord] = []
        self.trading_capital = cfg.initial_equity
        self.total_value = cfg.initial_equity
        self.floating_pnl = 0.0

        # 抽水机制
        self.waterfall = ProfitWaterfall(
            initial_capital=cfg.initial_equity,
            profit_share=cfg.waterfall_profit_share,
            growth_threshold=cfg.waterfall_growth_threshold
        )

        # 资金状态
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.current_day = None
        self.peak_equity = cfg.initial_equity  # 用于计算回撤

        # 复利相关
        self.compound_returns = []  # 记录每笔交易的收益率
        self.equity_at_entry = None  # 记录每笔交易开仓时的资金

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
        if period <= 0:
            return series * np.nan
        half = int(period / 2)
        sqrt_p = int(np.sqrt(period))
        wma1 = self._wma(series, half)
        wma2 = self._wma(series, period)
        diff = 2 * wma1 - wma2
        return self._wma(diff, sqrt_p)

    def _calc_indicators(self):
        df5 = self.df_5m
        df1 = self.df_1h

        # 1H 趋势 HMA
        df1["hma_fast_1h"] = self._hma(df1["close"], self.cfg.hma_fast_period_1h)
        df1["hma_slow_1h"] = self._hma(df1["close"], self.cfg.hma_slow_period_1h)
        df1["hma_dir_1h"] = np.where(
            df1["hma_fast_1h"] > df1["hma_slow_1h"], 1,
            np.where(df1["hma_fast_1h"] < df1["hma_slow_1h"], -1, 0)
        )

        # 4H 趋势 HMA
        df1_4h = df1.resample("4H").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()
        df1_4h["hma_fast_4h"] = self._hma(df1_4h["close"], self.cfg.hma_fast_period_4h)
        df1_4h["hma_slow_4h"] = self._hma(df1_4h["close"], self.cfg.hma_slow_period_4h)
        df1_4h["hma_dir_4h"] = np.where(
            df1_4h["hma_fast_4h"] > df1_4h["hma_slow_4h"], 1,
            np.where(df1_4h["hma_fast_4h"] < df1_4h["hma_slow_4h"], -1, 0)
        )

        # ADX
        adx_1h = talib.ADX(df1["high"], df1["low"], df1["close"], timeperiod=self.cfg.adx_period_1h)
        df1["adx_1h"] = adx_1h

        # ATR（1H）
        atr_1h = talib.ATR(df1["high"], df1["low"], df1["close"], timeperiod=14)
        df1["atr_1h"] = atr_1h

        # 合并 1H -> 5m
        df5 = df5.join(df1[["hma_dir_1h", "adx_1h", "atr_1h"]], how="left")
        df5["hma_dir_1h"].fillna(method="ffill", inplace=True)
        df5["adx_1h"].fillna(method="ffill", inplace=True)
        df5["atr_1h"].fillna(method="ffill", inplace=True)

        # 合并 4H -> 5m
        df5 = df5.join(df1_4h[["hma_dir_4h"]], how="left")
        df5["hma_dir_4h"].fillna(method="ffill", inplace=True)

        # BOLL（5m）
        df5["boll_mid_5m"], df5["boll_up_5m"], df5["boll_low_5m"] = talib.BBANDS(
            df5["close"],
            timeperiod=self.cfg.boll_period_5m,
            nbdevup=self.cfg.boll_std_5m,
            nbdevdn=self.cfg.boll_std_5m,
            matype=talib.MA_Type.SMA
        )

        # 5m EMA 快慢线
        df5["ema_fast_5m"] = talib.EMA(df5["close"], timeperiod=20)
        df5["ema_slow_5m"] = talib.EMA(df5["close"], timeperiod=50)

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(df5["close"], 12, 26, 9)
        df5["macd_hist_5m"] = macd_hist

        # 趋势方向 + 强度
        def calc_trend(row):
            dir_1h = row["hma_dir_1h"]
            dir_4h = row["hma_dir_4h"]
            adx_v = row["adx_1h"]

            if dir_1h == 0 or dir_4h == 0:
                return 0, 0

            if dir_1h == dir_4h:
                if adx_v < 20:
                    strength = 0
                elif adx_v < 25:
                    strength = 1
                else:
                    strength = 2
                return dir_1h, strength
            else:
                return 0, 0

        trend_dir = []
        trend_strength = []
        for _, r in df5.iterrows():
            d, s = calc_trend(r)
            trend_dir.append(d)
            trend_strength.append(s)
        df5["trend_dir"] = trend_dir
        df5["trend_strength"] = trend_strength

        self.df_5m = df5

    # ========= 数据加载 =========
    def load_data(self):
        df_5m = pd.read_parquet(self.cfg.data_5m_path)
        df_1h = pd.read_parquet(self.cfg.data_1h_path)

        df_5m = df_5m.sort_index()
        df_1h = df_1h.sort_index()

        self.df_5m = df_5m
        self.df_1h = df_1h

    # ========= 仓位 & SL/TP 计算 =========
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
            max_bars = cfg.max_bars_strong
        else:
            rr = 3.0
            sl_mult = 2.5
            max_bars = cfg.max_bars_normal

        # 止损价 & 止盈价
        sl_dist_abs = atr_1h * sl_mult
        if sl_dist_abs <= 0:
            return 0.0, 0.0, 0.0, 0, 0.0, 0.0

        if side == Side.LONG:
            stop_price = entry_price - sl_dist_abs
            take_price = entry_price + sl_dist_abs * rr
        else:
            stop_price = entry_price + sl_dist_abs
            take_price = entry_price - sl_dist_abs * rr

        if stop_price <= 0 or take_price <= 0:
            return 0.0, 0.0, 0.0, 0, 0.0, 0.0

        # 根据风险预算反推名义仓位
        risk_target = equity * cfg.risk_per_trade
        stop_dist_pct = abs(entry_price - stop_price) / entry_price
        if stop_dist_pct <= 0:
            return 0.0, 0.0, 0.0, 0, 0.0, 0.0

        notional_target = risk_target / stop_dist_pct

        # 仓位上限（杠杆只作为上限，不主动放大风险）
        pos_cap = cfg.position_cap
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

        side = "FLAT"
        trade_type = None
        entry_price = 0.0
        notional = 0.0
        stop_price = 0.0
        take_price = 0.0
        entry_time = None
        bars_held = 0
        atr_used_for_trade = 0.0
        rr_used_for_trade = 0.0
        trend_dir_for_trade = 0
        trend_strength_for_trade = 0

        self.trading_capital = self.cfg.initial_equity
        self.total_value = self.cfg.initial_equity
        self.floating_pnl = 0.0
        self.peak_equity = self.cfg.initial_equity
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.current_day = None
        self.compound_returns = []
        self.equity_at_entry = None

        for ts, row in df.iterrows():
            price_open = row["open"]
            price_high = row["high"]
            price_low = row["low"]
            price_close = row["close"]

            current_day = ts.date()
            if self.current_day != current_day:
                self.daily_pnl = 0.0
                self.current_day = current_day

            trend_dir_now = int(row["trend_dir"])
            trend_strength_now = int(row["trend_strength"])
            atr_1h = row["atr_1h"]
            mid5 = row["boll_mid_5m"]
            up5 = row["boll_up_5m"]
            low5 = row["boll_low_5m"]

            # === 1) 持仓管理 ===
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

                # 时间止盈/止损
                bars_held += 1
                if exit_price is None and bars_held >= (
                    self.cfg.max_bars_strong if trend_strength_for_trade >= 2 else self.cfg.max_bars_normal
                ):
                    exit_price = price_close
                    exit_reason = "TIME"

                # 趋势反转离场
                if exit_price is None:
                    if (trend_dir_for_trade == 1 and trend_dir_now <= 0) or \
                       (trend_dir_for_trade == -1 and trend_dir_now >= 0):
                        exit_price = price_close
                        exit_reason = "TREND_REV"

                if exit_price is not None:
                    # 计算实际盈亏（包含手续费和滑点）
                    if side == "LONG":
                        gross_pnl = notional * (exit_price - entry_price) / entry_price
                    else:
                        gross_pnl = notional * (entry_price - exit_price) / entry_price

                    # 计算手续费（开仓和平仓各一次）
                    fee_open = notional * self.cfg.fee_rate
                    fee_close = notional * self.cfg.fee_rate

                    # 计算滑点成本（入场和出场都有滑点）
                    slippage_cost = notional * self.cfg.slippage * 2

                    # 净盈亏
                    pnl = gross_pnl - fee_open - fee_close - slippage_cost

                    # 更新资金
                    self.trading_capital += pnl
                    self.total_value = self.waterfall.get_total_value()

                    # 记录复利收益率（基于开仓时的资金）
                    if self.equity_at_entry and self.equity_at_entry > 0:
                        trade_return_pct = pnl / self.equity_at_entry
                        self.compound_returns.append(trade_return_pct)
                    self.equity_at_entry = None

                    # 更新风控状态
                    self.daily_pnl += pnl
                    if pnl <= 0:
                        self.consecutive_losses += 1
                    else:
                        self.consecutive_losses = 0

                    self.peak_equity = max(self.peak_equity, self.total_value)
                    dd = (self.total_value - self.peak_equity) / self.peak_equity

                    # 记录交易
                    self.trades.append(
                        TradeRecord(
                            entry_time=entry_time,
                            exit_time=ts,
                            side=side,
                            trade_type=trade_type,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            notional=notional,
                            pnl=pnl,
                            rr=rr_used_for_trade,
                            bars_held=bars_held,
                            trend_strength=trend_strength_for_trade,
                            trend_dir=trend_dir_for_trade,
                            atr_1h=atr_used_for_trade,
                            exit_reason=exit_reason,
                        )
                    )

                    # 抽水机制
                    if self.cfg.enable_waterfall:
                        self.trading_capital = self.waterfall.update_after_trade(self.trading_capital)
                        self.total_value = self.waterfall.get_total_value()

                    # 风控终止条件
                    if self.consecutive_losses >= self.cfg.max_consecutive_losses:
                        print("触发最大连续亏损限制，提前停止策略。")
                        break

                    if self.daily_pnl <= -self.cfg.initial_equity * self.cfg.max_daily_loss_pct:
                        print("触发单日最大亏损限制，提前停止策略。")
                        break

                    if dd <= -self.cfg.max_drawdown_pct:
                        print("触发最大回撤限制，提前停止策略。")
                        break

                    # 平仓后重置持仓
                    side = "FLAT"
                    trade_type = None
                    entry_price = 0.0
                    notional = 0.0
                    stop_price = 0.0
                    take_price = 0.0
                    entry_time = None
                    bars_held = 0
                    atr_used_for_trade = 0.0
                    rr_used_for_trade = 0.0
                    trend_dir_for_trade = 0
                    trend_strength_for_trade = 0

            # === 2) 开仓信号 ===
            if side == "FLAT":
                hour = ts.hour
                if not (self.cfg.trading_start_hour <= hour < self.cfg.trading_end_hour):
                    continue

                trend_dir_now = int(row["trend_dir"])
                trend_strength_now = int(row["trend_strength"])
                if trend_dir_now == 0 or trend_strength_now <= 0:
                    continue

                # BOLL 相对位置
                if np.isnan(mid5) or np.isnan(up5) or np.isnan(low5):
                    continue
                if up5 == low5:
                    continue

                pos_in_band = (price_close - low5) / (up5 - low5)

                # 15m 的 K 线起点入场
                if ts.minute % 15 != 0:
                    continue

                # EMA/MACD 作为触发
                ema_fast = row["ema_fast_5m"]
                ema_slow = row["ema_slow_5m"]
                macd_hist = row["macd_hist_5m"]
                if any(np.isnan([ema_fast, ema_slow, macd_hist])):
                    continue

                sig_side = None
                sig_type = None

                if trend_dir_now > 0:
                    # 多头区间限制
                    if not (0.30 <= pos_in_band <= 0.85):
                        continue

                    # EMA 金叉 or MACD 柱由负转正
                    if ema_fast > ema_slow and macd_hist > 0:
                        sig_side = "LONG"
                        sig_type = TradeType.TREND

                elif trend_dir_now < 0:
                    # 空头区间限制
                    if not (0.15 <= pos_in_band <= 0.70):
                        continue

                    if ema_fast < ema_slow and macd_hist < 0:
                        sig_side = "SHORT"
                        sig_type = TradeType.TREND

                if sig_side is not None and sig_type is not None and atr_1h > 0:
                    # 入场价格（不在价格上加滑点，滑点统一在成本上结算）
                    if sig_side == "LONG":
                        entry_price = price_close
                    else:
                        entry_price = price_close

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

                        # 记录开仓时的资金，用于精确复利统计
                        self.equity_at_entry = self.trading_capital

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

        if plot:
            self._plot_equity_curve()

    # ========= 绩效汇总 =========
    def summary(self):
        trades = self.trades
        if not trades:
            print("无交易记录")
            return

        total_pnl = sum(t.pnl for t in trades)
        final_equity = self.total_value
        total_return = (final_equity / self.cfg.initial_equity - 1) * 100

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / len(trades) if trades else 0

        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        rr_real = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan

        # 用 daily equity 计算最大回撤（简化）
        eq_list = [self.cfg.initial_equity]
        cur_eq = self.cfg.initial_equity
        peak_eq = cur_eq
        max_dd = 0.0

        for t in trades:
            cur_eq += t.pnl
            peak_eq = max(peak_eq, cur_eq)
            dd = (cur_eq - peak_eq) / peak_eq
            max_dd = min(max_dd, dd)
            eq_list.append(cur_eq)

        print("=" * 45)
        print(f"===== V31 规则版 Teacher_V9 · 专业趋势跟随 回测结果 =====")
        print(f"交易对: {self.cfg.symbol}")
        print(f"初始资金: {self.cfg.initial_equity:,.2f}")
        print(f"最终资金: {final_equity:,.2f}")
        print(f"总收益率: {total_return:.2f}%")
        print(f"最大回撤: {max_dd*100:.2f}%")
        print(f"交易总笔数: {len(trades)}")
        print(f"胜率: {win_rate*100:.2f}%")
        print(f"平均盈利: ${avg_win:.2f}")
        print(f"平均亏损: ${avg_loss:.2f}")
        print(f"实际盈亏比: {rr_real:.2f}")
        print(f"手续费率: {self.cfg.fee_rate*100:.4f}%")
        print(f"滑点: {self.cfg.slippage*100:.4f}%")

        # 复利效果分析：基于每笔 pnl / 开仓资金 的收益率
        if self.cfg.compound_mode and self.compound_returns:
            compound_final = self.cfg.initial_equity
            simple_final = self.cfg.initial_equity
            for ret in self.compound_returns:
                compound_final *= (1 + ret)
                simple_final += ret * self.cfg.initial_equity

            compound_return = (compound_final / self.cfg.initial_equity - 1) * 100
            simple_return = (simple_final / self.cfg.initial_equity - 1) * 100
            print(f"\n复利效果分析:")
            print(f"简单累计收益: {simple_return:.2f}%")
            print(f"复利累计收益: {compound_return:.2f}%")
            print(f"复利增益: {compound_return - simple_return:.2f}%")

        # 抽水机制统计
        if self.cfg.enable_waterfall:
            print("\n" + "-" * 40)
            print("抽水机制统计")
            print("-" * 40)
            print(f"交易资金: {self.waterfall.trading_capital:,.2f}")
            print(f"利润池: {self.waterfall.profit_pool:,.2f}")
            print(f"抽水总额: {self.waterfall.total_water_drawn:,.2f}")
            print(f"抽水次数: {self.waterfall.water_draw_times}")

        print("=" * 45)

    # ========= 曲线图 =========
    def _plot_equity_curve(self):
        if not self.trades:
            return
        eq = [self.cfg.initial_equity]
        cur = self.cfg.initial_equity
        for t in self.trades:
            cur += t.pnl
            eq.append(cur)
        plt.figure(figsize=(12, 6))
        plt.plot(eq)
        plt.title("Equity Curve")
        plt.xlabel("Trades")
        plt.ylabel("Equity")
        plt.grid(True)
        plt.show()


# ========= 命令行入口 =========
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--export", type=str, default="")
    args = parser.parse_args()

    cfg = Config(symbol=args.symbol)
    engine = V31TrendSystemTeacherV9(cfg)
    engine.run_backtest(plot=not args.no_plot)
    engine.summary()

    if args.export:
        import csv

        with open(args.export, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "entry_time", "exit_time", "side", "trade_type",
                "entry_price", "exit_price", "notional", "pnl",
                "rr", "bars_held", "trend_strength", "trend_dir",
                "atr_1h", "exit_reason"
            ])
            for t in engine.trades:
                writer.writerow([
                    t.entry_time, t.exit_time, t.side, t.trade_type,
                    t.entry_price, t.exit_price, t.notional, t.pnl,
                    t.rr, t.bars_held, t.trend_strength, t.trend_dir,
                    t.atr_1h, t.exit_reason
                ])
        print(f"交易明细已导出到: {args.export}")
