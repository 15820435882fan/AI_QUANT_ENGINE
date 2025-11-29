#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
smart_backtest.py  (稳定版)

说明：
- 支持 baseline / ai_prod 两种“引擎”，目前主要推荐 ai_prod；
- 内置 5m 级别的模拟 K 线生成器，已经做了防“价格爆炸 / 溢出”处理；
- 资金管理、防爆仓逻辑做了多重安全阈值；
- 带有利润抽取（进保险柜）与 AI 评分系统。
"""

import argparse
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    # 可选真实数据模块（如果没有，会自动降级为模拟数据）
    from real_market_data import load_for_smart_backtest
except Exception:  # noqa: E722
    load_for_smart_backtest = None


# ============================================================
# 日志配置
# ============================================================
logger = logging.getLogger("SmartBacktest")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


# ============================================================
# 工具函数 & 数据生成
# ============================================================

def parse_symbols(sym_str: str) -> List[str]:
    return [s.strip() for s in sym_str.split(",") if s.strip()]


def base_price_for_symbol(symbol: str) -> float:
    base = symbol.upper().split("/")[0]
    mapping = {
        "BTC": 30000.0,
        "ETH": 1500.0,
        "SOL": 30.0,
        "BNB": 300.0,
        "XRP": 0.6,
    }
    return mapping.get(base, 100.0)


def simulate_market_data(
    symbol: str,
    days: int,
    interval_minutes: int = 5,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    较为“稳健”的 5m 随机行情模拟器，专门做了防溢出处理：
    - 使用对数收益（log-return）叠加；
    - 控制单步波动 & 总体波动；
    - 对价格做上下边界夹紧（clip），避免爆炸。
    """
    rng = np.random.default_rng(seed)
    steps_per_day = int(24 * 60 / interval_minutes)
    n = max(steps_per_day * days, 100)

    base_price = base_price_for_symbol(symbol)
    # 日波动率设置在 4% 左右
    daily_vol = 0.04
    # 每步波动率（5m）
    step_vol = daily_vol / math.sqrt(steps_per_day)

    # 加一点轻微上升 drift（年化 ~20% 左右的量级）
    annual_drift = 0.20
    daily_drift = annual_drift / 365.0
    step_drift = daily_drift / steps_per_day

    # 生成 log-return
    eps = rng.normal(loc=0.0, scale=step_vol, size=n)
    # 控制极端：单步收益不要超过 ±20%
    eps = np.clip(eps, -0.2, 0.2)
    log_returns = step_drift + eps

    # 价格路径：log_price(t) = log(P0) + cumsum(log_returns)
    log_p0 = math.log(base_price)
    log_price_path = log_p0 + np.cumsum(log_returns)
    price_path = np.exp(log_price_path)

    # 再做一次全局 clip，防止极端爆炸
    lower = base_price * 0.3
    upper = base_price * 5.0
    price_path = np.clip(price_path, lower, upper)

    # 简单构造 OHLCV
    close = price_path
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    # 高低价在收盘价上下浮动一个很小的范围
    hl_spread = rng.normal(loc=0.0, scale=0.002, size=n)
    high = close * (1.0 + np.abs(hl_spread))
    low = close * (1.0 - np.abs(hl_spread))
    # 保证 high >= max(open, close), low <= min(open, close)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = rng.uniform(10_000, 50_000, size=n)

    # 构造时间序列（倒推 days 天，间隔 5m）
    end = pd.Timestamp.utcnow().floor("min")
    index = pd.date_range(end=end, periods=n, freq=f"{interval_minutes}min")

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    return df


# ============================================================
# 策略引擎
# ============================================================

@dataclass
class Signal:
    side: int            # 1: 做多, -1: 做空, 0: 空仓/观望
    sl_pct: float        # 止损百分比（相对入场价）
    tp_pct: float        # 止盈百分比
    confidence: float    # 0~1
    reason: str          # 文字说明


class BaseEngine:
    name: str = "baseline"

    def __init__(
        self,
        fast_ma: int = 8,
        slow_ma: int = 21,
        up_th: float = 0.004,
        down_th: float = 0.004,
    ):
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.up_th = up_th
        self.down_th = down_th

    def _calc_trend_and_vol(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        close = df["close"]
        ema_fast = close.ewm(span=self.fast_ma, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_ma, adjust=False).mean()
        trend = ema_fast / ema_slow - 1.0

        # 近 48 根的波动率（5m * 48 ≈ 4 小时）
        vol = close.pct_change().rolling(48, min_periods=10).std()
        vol = vol.fillna(method="bfill").fillna(0.001)
        vol = vol.clip(0.001, 0.03)
        return trend, vol

    def generate_signals(self, df: pd.DataFrame) -> Dict[pd.Timestamp, Signal]:
        trend, vol = self._calc_trend_and_vol(df)
        signals: Dict[pd.Timestamp, Signal] = {}

        for ts, tr, v in zip(df.index, trend, vol):
            if tr > self.up_th:
                side = 1
                # 止损设为 2x 波动率，TP 为 3x
                sl_pct = float(max(0.003, min(0.03, 2.0 * v)))
                tp_pct = float(max(0.01, min(0.08, 3.0 * v)))
                conf = float(min(1.0, tr / (self.up_th * 2.0)))
                reason = f"上升趋势, trend={tr:.4f}, vol={v:.4f}"
            elif tr < -self.down_th:
                side = -1
                sl_pct = float(max(0.003, min(0.03, 2.0 * v)))
                tp_pct = float(max(0.01, min(0.08, 3.0 * v)))
                conf = float(min(1.0, abs(tr) / (self.down_th * 2.0)))
                reason = f"下降趋势, trend={tr:.4f}, vol={v:.4f}"
            else:
                side = 0
                sl_pct = 0.0
                tp_pct = 0.0
                conf = 0.0
                reason = "趋势弱, 观望"

            signals[ts] = Signal(side=side, sl_pct=sl_pct, tp_pct=tp_pct, confidence=conf, reason=reason)
        return signals


class AIProdEngine(BaseEngine):
    """
    “AI 大脑” 版本：在 Baseline 基础上增加了一些启发式判断，
    但仍然保持完全可解释 & 不使用黑箱模型。
    """

    name: str = "ai_prod"

    def __init__(self):
        super().__init__(fast_ma=7, slow_ma=24, up_th=0.003, down_th=0.003)

    def generate_signals(self, df: pd.DataFrame) -> Dict[pd.Timestamp, Signal]:
        trend, vol = self._calc_trend_and_vol(df)
        close = df["close"]

        # 布林带（中轨 = ema_slow, 宽度与 vol 挂钩）
        ema_mid = close.ewm(span=self.slow_ma, adjust=False).mean()
        band_width = (vol * 10).clip(0.5, 3.0)
        upper = ema_mid * (1 + band_width / 100)
        lower = ema_mid * (1 - band_width / 100)

        signals: Dict[pd.Timestamp, Signal] = {}

        for ts in df.index:
            tr = float(trend.loc[ts])
            v = float(vol.loc[ts])
            c = float(close.loc[ts])
            mid = float(ema_mid.loc[ts])
            up = float(upper.loc[ts])
            lo = float(lower.loc[ts])

            side = 0
            sl_pct = 0.0
            tp_pct = 0.0
            conf = 0.0
            reason = "观望"

            # ====== 做多 / 做空逻辑（简化版） ======
            # 1）价格突破中轨，且趋势配合
            if c > mid and tr > self.up_th:
                side = 1
                # 止损：略大于短期波动；止盈：大约 2~3 倍波动
                sl_pct = float(max(0.004, min(0.025, 1.8 * v)))
                tp_pct = float(max(0.012, min(0.07, 3.0 * v)))
                conf = float(min(1.0, (tr / (self.up_th * 2.0)) + (c - mid) / (mid * 0.01)))
                reason = f"趋势向上 & 价格在中轨上方, trend={tr:.4f}, vol={v:.4f}"
            elif c < mid and tr < -self.down_th:
                side = -1
                sl_pct = float(max(0.004, min(0.025, 1.8 * v)))
                tp_pct = float(max(0.012, min(0.07, 3.0 * v)))
                conf = float(min(1.0, (abs(tr) / (self.down_th * 2.0)) + (mid - c) / (mid * 0.01)))
                reason = f"趋势向下 & 价格在中轨下方, trend={tr:.4f}, vol={v:.4f}"
            else:
                # 2）布林带极值的“超跌反弹 / 超涨回落”尝试（信心较低）
                if c < lo and tr > -self.down_th:
                    side = 1
                    sl_pct = float(max(0.005, min(0.03, 2.0 * v)))
                    tp_pct = float(max(0.015, min(0.08, 3.5 * v)))
                    conf = 0.4
                    reason = f"触及下轨, 超跌反弹尝试, trend={tr:.4f}, vol={v:.4f}"
                elif c > up and tr < self.up_th:
                    side = -1
                    sl_pct = float(max(0.005, min(0.03, 2.0 * v)))
                    tp_pct = float(max(0.015, min(0.08, 3.5 * v)))
                    conf = 0.4
                    reason = f"触及上轨, 超涨回落尝试, trend={tr:.4f}, vol={v:.4f}"
                else:
                    side = 0
                    sl_pct = 0.0
                    tp_pct = 0.0
                    conf = 0.0
                    reason = "信号不明显, 观望"

            signals[ts] = Signal(side=side, sl_pct=sl_pct, tp_pct=tp_pct, confidence=conf, reason=reason)
        return signals


# ============================================================
# 回测核心
# ============================================================

@dataclass
class Trade:
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: int
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    reason: str


@dataclass
class RiskConfig:
    fixed_risk: float = 0.005     # 每笔风险占总权益比例（0.5%）
    max_leverage: float = 3.0     # 杠杆上限
    max_notional: float = 1_000_000.0  # 单笔名义仓位上限
    min_sl_pct: float = 0.003     # 止损下限（0.3%）
    max_sl_pct: float = 0.05      # 止损上限（5%）

    max_daily_loss: float = 0.08  # 单日最大亏损 8%
    max_consec_losses: int = 6    # 连续亏损 N 笔后暂停
    cool_off_bars: int = 288      # 冷静期长度（288 根 ~ 1 天）


@dataclass
class SkimConfig:
    trigger_pct: float = 0.10     # 账号从高点回看，盈利超过 10% 时触发
    skim_pct: float = 0.20        # 抽取 20% 盈利进“保险柜”


class SmartBacktest:
    def __init__(
        self,
        symbols: List[str],
        days: int,
        engine: str = "ai_prod",
        use_real_data: bool = False,
        risk_cfg: Optional[RiskConfig] = None,
        skim_cfg: Optional[SkimConfig] = None,
    ):
        self.symbols = symbols
        self.days = days
        self.use_real_data = use_real_data
        self.risk_cfg = risk_cfg or RiskConfig()
        self.skim_cfg = skim_cfg or SkimConfig()

        self.initial_equity = 10_000.0
        self.equity = self.initial_equity
        self.vault = 0.0  # “保险柜”里的安全利润（不可回吐）

        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.symbol_stats: Dict[str, Dict] = {}
        self.cool_off_until: Dict[str, pd.Timestamp] = {}
        self.consec_losses: Dict[str, int] = {}
        self.daily_pnl: Dict[pd.Timestamp, float] = {}

        if engine == "baseline":
            self.engine = BaseEngine()
        else:
            self.engine = AIProdEngine()
        self.engine_name = self.engine.name

        logger.info("🚀 智能回测系统初始化完成")
        logger.info(
            "💰 初始资金: $%.2f, 杠杆: %.1fx, 使用真实数据: %s, 引擎: %s",
            self.initial_equity,
            self.risk_cfg.max_leverage,
            self.use_real_data,
            self.engine_name,
        )

    # ------------------ 数据获取 ------------------

    def _load_data_for_symbol(self, symbol: str) -> pd.DataFrame:
        if self.use_real_data and load_for_smart_backtest is not None:
            try:
                df = load_for_smart_backtest(symbol, days=self.days, interval="5m")
                if df is not None and not df.empty:
                    logger.info("📊 使用真实市场数据: %s (%d 行)", symbol, len(df))
                    return df
            except Exception as e:  # noqa: E722
                print(f"❌ 下载真实数据失败: {e}")
                print(f"⚠️ 使用 fallback 模拟数据: {symbol}")

        df = simulate_market_data(symbol, days=self.days, interval_minutes=5)
        logger.info("📊 使用模拟市场数据: %s (%d 行)", symbol, len(df))
        return df

    # ------------------ 资金管理 ------------------

    def _update_equity_curve(self, ts: pd.Timestamp):
        """在每根 K 线末尾记录一次权益，顺便做 NaN/Inf 防护。"""
        eq = float(self.equity)
        if not np.isfinite(eq) or eq <= 0:
            # 若出现异常，强制清零，并停止后续增长
            eq = max(eq, 0.0)
        self.equity_curve.append(eq)

    def _skim_profits(self):
        """当权益突破新高且盈利超过触发阈值时，抽取一部分利润进保险柜。"""
        if not self.equity_curve:
            return
        eq = self.equity_curve[-1]
        if not np.isfinite(eq) or eq <= 0:
            return

        peak = max(self.equity_curve)
        if peak <= 0:
            return

        # 当前已经有多少“浮动盈利”
        float_profit = eq - self.initial_equity
        if float_profit <= 0:
            return

        # 相对整体初始资金的盈利比例
        total_gain = (eq / self.initial_equity) - 1.0

        # 只要整体盈利超过 trigger_pct，就允许抽取
        if total_gain >= self.skim_cfg.trigger_pct:
            skim_amount = float(float_profit * self.skim_cfg.skim_pct)
            skim_amount = max(0.0, min(skim_amount, eq * 0.3))  # 最多抽 30% 资金
            if skim_amount > 0:
                self.equity -= skim_amount
                self.vault += skim_amount
                logger.info(
                    "💰 触发利润抽取: 抽取 %.2f 至保险柜，当前保险柜余额=%.2f, 账户权益=%.2f",
                    skim_amount,
                    self.vault,
                    self.equity,
                )

    # ------------------ 订单执行 ------------------

    def _run_single_symbol(self, symbol: str):
        df = self._load_data_for_symbol(symbol)
        signals = self.engine.generate_signals(df)

        pos_side = 0
        pos_qty = 0.0
        pos_entry_price = 0.0
        pos_sl_price = 0.0
        pos_tp_price = 0.0

        wins = 0
        losses = 0
        trade_count = 0
        skimmed_for_symbol = 0.0

        self.consec_losses.setdefault(symbol, 0)
        self.cool_off_until.setdefault(symbol, df.index[0])

        for ts, row in df.iterrows():
            price = float(row["close"])

            # 记录日度 PnL（简化：每根 k 线都记录为 0，只有平仓时才更新）
            d = ts.normalize()
            self.daily_pnl.setdefault(d, 0.0)

            # 冷静期：直接观望
            if ts < self.cool_off_until[symbol]:
                self._update_equity_curve(ts)
                continue

            sig = signals.get(ts, Signal(0, 0.0, 0.0, 0.0, "无信号"))

            # 如果当前有持仓，先检查止盈/止损
            if pos_side != 0 and pos_qty > 0:
                exit_reason = None
                exit_price = price

                # 多头
                if pos_side > 0:
                    if price <= pos_sl_price:
                        exit_reason = "止损"
                        exit_price = pos_sl_price
                    elif price >= pos_tp_price:
                        exit_reason = "止盈"
                        exit_price = pos_tp_price
                else:  # 空头
                    if price >= pos_sl_price:
                        exit_reason = "止损"
                        exit_price = pos_sl_price
                    elif price <= pos_tp_price:
                        exit_reason = "止盈"
                        exit_price = pos_tp_price

                if exit_reason is not None:
                    pnl = (exit_price - pos_entry_price) * pos_qty * pos_side
                    self.equity += pnl
                    self.trades.append(
                        Trade(
                            symbol=symbol,
                            entry_time=None,  # 简化：不记录；如有需要可拓展
                            exit_time=ts,
                            side=pos_side,
                            entry_price=pos_entry_price,
                            exit_price=exit_price,
                            qty=pos_qty,
                            pnl=pnl,
                            reason=exit_reason,
                        )
                    )
                    self.daily_pnl[d] += pnl
                    trade_count += 1

                    if pnl >= 0:
                        wins += 1
                        self.consec_losses[symbol] = 0
                    else:
                        losses += 1
                        self.consec_losses[symbol] += 1

                    # 日内风控：最大亏损
                    day_loss = self.daily_pnl[d]
                    if day_loss < -self.initial_equity * self.risk_cfg.max_daily_loss:
                        # 当天亏损超限，本币种冷静一整天
                        self.cool_off_until[symbol] = ts + pd.Timedelta(
                            minutes=5 * self.risk_cfg.cool_off_bars
                        )
                        logger.info(
                            "🧊 %s 单日亏损超限，进入冷静期至 %s",
                            symbol,
                            self.cool_off_until[symbol],
                        )

                    # 连续亏损风控
                    if self.consec_losses[symbol] >= self.risk_cfg.max_consec_losses:
                        self.cool_off_until[symbol] = ts + pd.Timedelta(
                            minutes=5 * self.risk_cfg.cool_off_bars
                        )
                        self.consec_losses[symbol] = 0
                        logger.info(
                            "🧊 %s 连续亏损 %d 笔，进入冷静期至 %s",
                            symbol,
                            self.risk_cfg.max_consec_losses,
                            self.cool_off_until[symbol],
                        )

                    # 平仓后，清空仓位
                    pos_side = 0
                    pos_qty = 0.0
                    pos_entry_price = 0.0
                    pos_sl_price = 0.0
                    pos_tp_price = 0.0

                    # 平仓后尝试抽取利润
                    before_vault = self.vault
                    self._skim_profits()
                    skimmed_for_symbol += (self.vault - before_vault)

            # 若当前无仓位，可以考虑开仓
            if pos_side == 0 and sig.side != 0 and sig.confidence > 0:
                # 资金安全检查
                eq = max(0.0, float(self.equity))
                if not np.isfinite(eq) or eq <= 0:
                    logger.warning("⚠️ 权益异常，停止开新仓: equity=%.4f", eq)
                    self._update_equity_curve(ts)
                    continue

                # 计算每笔风险金额
                risk_amount = eq * self.risk_cfg.fixed_risk
                sl_pct = float(
                    min(
                        max(sig.sl_pct, self.risk_cfg.min_sl_pct),
                        self.risk_cfg.max_sl_pct,
                    )
                )
                if sl_pct <= 0:
                    self._update_equity_curve(ts)
                    continue

                # 名义仓位：风险金额 / 止损距离
                notional = risk_amount / sl_pct
                # 乘杠杆上限
                max_notional = eq * self.risk_cfg.max_leverage
                notional = min(notional, max_notional, self.risk_cfg.max_notional)

                if notional <= 0:
                    self._update_equity_curve(ts)
                    continue

                qty = notional / price
                if qty <= 0 or not np.isfinite(qty):
                    self._update_equity_curve(ts)
                    continue

                pos_side = sig.side
                pos_qty = qty
                pos_entry_price = price

                if pos_side > 0:
                    pos_sl_price = price * (1.0 - sl_pct)
                    pos_tp_price = price * (1.0 + sig.tp_pct)
                else:
                    pos_sl_price = price * (1.0 + sl_pct)
                    pos_tp_price = price * (1.0 - sig.tp_pct)

            # 记录权益
            self._update_equity_curve(ts)

        # 记录每个 symbol 的统计数据
        self.symbol_stats[symbol] = {
            "trades": trade_count,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / trade_count * 100.0) if trade_count > 0 else 0.0,
            "skimmed": skimmed_for_symbol,
        }

    # ------------------ 绩效评估 ------------------

    def _compute_max_drawdown(self) -> float:
        eq = np.asarray(self.equity_curve, dtype=float)
        mask = np.isfinite(eq) & (eq > 0)
        if mask.sum() < 2:
            return 1.0  # 100% 回撤（极端保守）

        eq = eq[mask]
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        max_dd = float(dd.min())
        return abs(max_dd)

    def _compute_sharpe_like(self) -> float:
        eq = np.asarray(self.equity_curve, dtype=float)
        mask = np.isfinite(eq) & (eq > 0)
        if mask.sum() < 2:
            return 0.0

        eq = eq[mask]
        rets = np.diff(eq) / eq[:-1]
        if len(rets) < 2:
            return 0.0
        mu = float(np.mean(rets))
        sigma = float(np.std(rets, ddof=1))
        if sigma <= 0 or not np.isfinite(sigma):
            return 0.0

        # 以 5m 为单位，粗略折算成年化
        bars_per_day = 24 * 60 / 5
        days_per_year = 365
        scale = math.sqrt(bars_per_day * days_per_year)
        sharpe_like = (mu / sigma) * scale
        return sharpe_like

    def _ai_score(self, final_equity: float, max_dd: float, sharpe_like: float) -> Tuple[float, str, str]:
        """
        返回 (score, grade, comment)
        分数越高越好；50 分以上才勉强值得小仓位尝试。
        """
        if not np.isfinite(final_equity) or final_equity <= 0:
            return 5.0, "E", "回测结果异常，净值无效。"

        ret = max(0.0, final_equity / self.initial_equity - 1.0)

        # 1）收益部分（最多 60 分）
        if ret <= 0:
            score_ret = 0.0
        elif ret < 1:
            score_ret = ret * 30.0  # 100% 收益给 30 分
        elif ret < 5:
            score_ret = 30.0 + (ret - 1.0) / 4.0 * 20.0  # 5x 给 50 分
        else:
            score_ret = 55.0  # 非常高的收益但不继续线性加分

        # 2）回撤部分（最多 25 分）
        if max_dd <= 0.1:
            score_dd = 25.0
        elif max_dd <= 0.2:
            score_dd = 18.0
        elif max_dd <= 0.3:
            score_dd = 10.0
        elif max_dd <= 0.5:
            score_dd = 5.0
        else:
            score_dd = 0.0

        # 3）Sharpe-like（最多 15 分）
        if not np.isfinite(sharpe_like) or sharpe_like <= 0:
            score_sh = 0.0
        elif sharpe_like < 1:
            score_sh = 5.0
        elif sharpe_like < 2:
            score_sh = 10.0
        else:
            score_sh = 15.0

        score = score_ret + score_dd + score_sh
        score = float(max(0.0, min(100.0, score)))

        if score >= 80:
            grade = "A"
            comment = "高收益且回撤可控，适合在严格风控前提下小规模实盘试验。"
        elif score >= 65:
            grade = "B"
            comment = "收益不错，但回撤或波动偏大，需要进一步精细化风控后再考虑实盘。"
        elif score >= 50:
            grade = "C"
            comment = "策略风险收益比一般，建议先小仓位或仅用作研究参考。"
        elif score >= 35:
            grade = "D"
            comment = "策略表现偏弱，暂不建议用于真实资金，仅供研究。"
        else:
            grade = "E"
            comment = "策略质量较差或结果异常，不建议使用。"

        return score, grade, comment

    # ------------------ 主流程 ------------------

    def run(self):
        logger.info(
            "🎯 开始智能回测: %s, 天数=%d",
            self.symbols,
            self.days,
        )

        for sym in self.symbols:
            logger.info("🔍 测试币种: %s", sym)
            self._run_single_symbol(sym)

        final_equity = float(self.equity)
        max_dd = self._compute_max_drawdown()
        sharpe_like = self._compute_sharpe_like()

        logger.info("")
        logger.info("=" * 80)
        logger.info("🧠 智能量化交易系统 - 回测报告")
        logger.info("=" * 80)
        logger.info("")
        logger.info("📈 智能性能汇总:")
        logger.info("  测试币种: %d 个", len(self.symbols))
        logger.info("  总交易次数: %d 笔", sum(s["trades"] for s in self.symbol_stats.values()))
        logger.info("  总收益: $%.2f", final_equity - self.initial_equity)
        logger.info("  最终资金: $+%.2f", final_equity)
        avg_win_rate = (
            np.mean([s["win_rate"] for s in self.symbol_stats.values()])
            if self.symbol_stats
            else 0.0
        )
        logger.info("  平均胜率: %.1f%%", avg_win_rate)
        logger.info("  最大回撤: %.1f%%", max_dd * 100.0)
        logger.info("  简易 Sharpe 类指标: %.2f", sharpe_like)

        # 抽佣统计
        total_skimmed = sum(s["skimmed"] for s in self.symbol_stats.values())
        logger.info("")
        logger.info("📊 各币种智能表现:")
        for sym, st in self.symbol_stats.items():
            logger.info(
                "  🟡 %s: %d 笔, 胜率: %.1f%%, 抽取到保险柜: $%.2f",
                sym,
                st["trades"],
                st["win_rate"],
                st["skimmed"],
            )

        logger.info("")
        logger.info("🏦 利润抽取 + 复利模拟（简化版）:")
        logger.info(
            "  回测期间共抽取到“保险柜”的安全利润≈$%.2f；若将这些视作完全风险隔离的收益，剩余资金继续用于复利。",
            total_skimmed,
        )

        # AI 评分
        score, grade, comment = self._ai_score(final_equity, max_dd, sharpe_like)
        logger.info("")
        logger.info("🤖 AI 风险收益评分:")
        logger.info("  综合得分: %.1f / 100, 等级: %s, 评语: %s", score, grade, comment)
        logger.info("")
        logger.info("🎉 智能回测完成！")
        logger.info("=" * 80)


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="智能量化回测系统（SmartBacktest 稳定版）")
    parser.add_argument(
        "--symbols",
        type=str,
        required=True,
        help="逗号分隔的交易对列表，例如: BTC/USDT,ETH/USDT",
    )
    parser.add_argument("--days", type=int, default=30, help="回测天数（默认 30）")
    parser.add_argument(
        "--engine",
        type=str,
        choices=["baseline", "ai_prod"],
        default="ai_prod",
        help="策略引擎类型（默认 ai_prod）",
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="尝试使用 real_market_data.load_for_smart_backtest 作为真实行情（若失败会自动回退模拟数据）",
    )

    args = parser.parse_args()
    symbols = parse_symbols(args.symbols)

    backtest = SmartBacktest(
        symbols=symbols,
        days=args.days,
        engine=args.engine,
        use_real_data=args.use_real_data,
    )
    backtest.run()


if __name__ == "__main__":
    main()
