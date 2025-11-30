#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SmartBacktest v8_1
===============================
第二季 · Step3.5：分币种参数 + 动态策略权重 + 因子打分版

特性：
- 真实 Binance 数据 + fallback 模拟
- 5 大策略信号（MACD / EMA / Turtle / BOLL / Breakout）
- 动态策略权重（基于过去窗口的方向有效性）
- 趋势强度因子 trend_strength
- Entry Score 打分开仓（趋势 + 策略共识）
- ATR 止损 / 止盈 + Trailing Stop
- 分币种风险参数（BTC / ETH 有不同 Risk Profile）
"""

# ============================================================
# 0. 禁用代理，避免请求被本地代理劫持
# ============================================================
import os
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

# ============================================================
# 基础库
# ============================================================
import argparse
import logging
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from real_market_data import RealMarketData

# ============================================================
# 日志
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. 模拟 K 线（稳定版）
# ============================================================
def generate_mock_data(symbol: str, days: int = 30, seed: Optional[int] = None) -> pd.DataFrame:
    """
    生成一个简易的随机 5m K 线，用于没有真实数据时的 fallback。
    """
    if seed is not None:
        np.random.seed(seed)

    periods = days * 24 * 12  # 5 分钟K线数量
    if periods <= 1:
        periods = 288

    prices = [100.0]
    for _ in range(periods):
        drift = np.random.normal(0, 1)
        prices.append(prices[-1] * (1 + drift * 0.001))
    prices = np.array(prices)

    openp = prices[:-1]
    closep = prices[1:]
    highp = np.maximum(openp, closep)
    lowp = np.minimum(openp, closep)
    vol = np.random.rand(periods) * 10

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=periods, freq="5min"),
            "open": openp,
            "high": highp,
            "low": lowp,
            "close": closep,
            "volume": vol,
        }
    )
    df.set_index("timestamp", inplace=True)
    return df


# ============================================================
# 2. 回测结果结构
# ============================================================
class SymbolResult:
    def __init__(self, pnl: float, trades: int, wins: int, max_dd_pct: float):
        self.pnl = pnl
        self.trades = trades
        self.wins = wins
        self.max_dd_pct = max_dd_pct

    @property
    def win_rate(self) -> float:
        return (self.wins / self.trades * 100) if self.trades > 0 else 0.0


# ============================================================
# 3. 基础指标：MA / RSI / ATR + 趋势强度
# ============================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # 均线
    d["ma_fast"] = d["close"].rolling(20).mean()
    d["ma_slow"] = d["close"].rolling(50).mean()

    d["trend_long_ok"] = d["ma_fast"] > d["ma_slow"]
    d["trend_short_ok"] = d["ma_fast"] < d["ma_slow"]

    # RSI
    delta = d["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    d["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    d["rsi_long_ok"] = d["rsi"] < 70
    d["rsi_short_ok"] = d["rsi"] > 30

    # ATR
    high_low = d["high"] - d["low"]
    high_close = (d["high"] - d["close"].shift(1)).abs()
    low_close = (d["low"] - d["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    d["tr"] = tr
    d["atr"] = d["tr"].rolling(14).mean()

    # 趋势强度：基于 EMA20 斜率
    ema20 = d["close"].ewm(span=20, adjust=False).mean()
    d["ema20"] = ema20
    slope = (ema20 - ema20.shift(5)) / (ema20.shift(5).abs() + 1e-9)
    # 趋势强度压缩到 [0,1]
    trend_strength = slope.abs() * 10.0  # 放大
    trend_strength = trend_strength.clip(lower=0.0, upper=1.0)
    d["trend_strength"] = trend_strength.fillna(0.0)

    return d


# ============================================================
# 4. 策略信号（MACD / EMA / Turtle / BOLL / Breakout）
# ============================================================
def compute_strategy_signals(d: pd.DataFrame) -> pd.DataFrame:
    df = d.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # MACD
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - macd_signal
    prev_hist = hist.shift(1)
    sig_macd = pd.Series(0, index=df.index, dtype=float)
    sig_macd[(prev_hist <= 0) & (hist > 0)] = 1
    sig_macd[(prev_hist >= 0) & (hist < 0)] = -1
    df["sig_macd"] = sig_macd

    # EMA 趋势策略
    ema_short = close.ewm(span=20, adjust=False).mean()
    ema_long = close.ewm(span=50, adjust=False).mean()
    sig_ema = pd.Series(0, index=df.index, dtype=float)
    sig_ema[ema_short > ema_long] = 1
    sig_ema[ema_short < ema_long] = -1
    df["sig_ema"] = sig_ema

    # Turtle 通道突破
    breakout_window = 20
    hh = high.rolling(window=breakout_window, min_periods=1).max()
    ll = low.rolling(window=breakout_window, min_periods=1).min()
    sig_turtle = pd.Series(0, index=df.index, dtype=float)
    sig_turtle[close > hh.shift(1)] = 1
    sig_turtle[close < ll.shift(1)] = -1
    df["sig_turtle"] = sig_turtle

    # Bollinger 收敛/扩散（逆势反转型）
    window = 20
    std = close.rolling(window=window, min_periods=1).std().fillna(0.0)
    ma = close.rolling(window=window, min_periods=1).mean()
    upper = ma + 2.0 * std
    lower = ma - 2.0 * std
    sig_boll = pd.Series(0, index=df.index, dtype=float)
    sig_boll[close < lower] = 1
    sig_boll[close > upper] = -1
    df["sig_boll"] = sig_boll

    # Breakout 策略（区间突破）
    lookback = 50
    threshold = 0.01
    rolling_max = close.rolling(window=lookback, min_periods=1).max()
    rolling_min = close.rolling(window=lookback, min_periods=1).min()
    sig_break = pd.Series(0, index=df.index, dtype=float)
    sig_break[close > rolling_max * (1 + threshold)] = 1
    sig_break[close < rolling_min * (1 - threshold)] = -1
    df["sig_break"] = sig_break

    return df


# ============================================================
# 5. 动态策略权重 + 合成信号
# ============================================================
def compute_dynamic_weights_and_ensemble(
    df: pd.DataFrame,
    window: int = 200,
    horizon: int = 3,
) -> pd.DataFrame:
    """
    对每个时间点 t：
      - 回看 [t-window, t-1] 上各策略信号对未来 horizon 根的方向预测效果
      - 得到各策略 accuracy
      - 转换成权重 w_i(t)
      - 计算 ensemble_raw(t) = Σ sig_i(t) * w_i(t)
      - ensemble_dir(t) = sign(ensemble_raw(t))
    """
    d = df.copy()
    close = d["close"].values
    sig_names = ["sig_macd", "sig_ema", "sig_turtle", "sig_boll", "sig_break"]
    sig_arrays = {name: d[name].values for name in sig_names}

    n = len(d)
    weights = {name: np.zeros(n, dtype=float) for name in sig_names}
    ensemble_raw = np.zeros(n, dtype=float)
    ensemble_dir = np.zeros(n, dtype=float)

    eps = 0.01  # 避免全零

    for i in range(n):
        start = max(0, i - window)
        end_j = i - horizon  # j+horizon < i

        if end_j <= start:
            # 历史数据不足：等权
            w_equal = 1.0 / len(sig_names)
            for name in sig_names:
                weights[name][i] = w_equal
        else:
            accs = []
            for name in sig_names:
                sig = sig_arrays[name]
                correct = 0
                total = 0
                for j in range(start, end_j):
                    s = sig[j]
                    if s == 0:
                        continue
                    if j + horizon >= i:
                        continue
                    ret = close[j + horizon] - close[j]
                    if s * ret > 0:
                        correct += 1
                    total += 1
                acc = correct / total if total > 0 else 0.0
                accs.append(acc)

            sumw = sum(a + eps for a in accs)
            if sumw <= 0:
                w_equal = 1.0 / len(sig_names)
                for name in sig_names:
                    weights[name][i] = w_equal
            else:
                for k, name in enumerate(sig_names):
                    weights[name][i] = (accs[k] + eps) / sumw

        # 当前时刻合成信号
        raw = 0.0
        for name in sig_names:
            raw += sig_arrays[name][i] * weights[name][i]
        ensemble_raw[i] = raw

        if abs(raw) < 0.1:
            ensemble_dir[i] = 0.0
        else:
            ensemble_dir[i] = 1.0 if raw > 0 else -1.0

    # 写回 DataFrame
    for name in sig_names:
        d[f"w_{name}"] = weights[name]
    d["ensemble_raw"] = ensemble_raw
    d["ensemble_dir"] = ensemble_dir

    return d


# ============================================================
# 6. 自适应信号引擎（V8_1）
# ============================================================
class AdaptiveSignalEngine:
    """
    V8_1 引擎：
    - 分币种参数（BTC / ETH 风险配置不同）
    - 多因子过滤（趋势 / RSI）
    - 动态策略权重 + 合成信号
    - Entry Score 打分
    - ATR 止损 / 止盈 + Trailing Stop
    - 连续亏损冷静期
    """

    def __init__(self):
        # 基础参数
        self.base_params = {
            "sl_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "trail_atr_mult": 1.5,
            "min_rr": 1.5,
            "risk_per_trade": 0.01,
            "max_loss_streak": 3,
            "cooldown_bars": 12 * 12,  # 12小时（5mK）
        }
        # 分币种 override（可继续扩展）
        self.symbol_overrides = {
            "ETH": {
                "sl_atr_mult": 2.0,
                "tp_atr_mult": 4.0,
                "trail_atr_mult": 2.0,
                "min_rr": 1.2,
                "risk_per_trade": 0.005,
                "max_loss_streak": 3,
                "cooldown_bars": 12 * 12,
            }
        }

    def _get_params_for_symbol(self, symbol: str) -> Dict[str, float]:
        params = self.base_params.copy()
        for key, override in self.symbol_overrides.items():
            if key in symbol.upper():
                params.update(override)
        return params

    def _build_filters(self, d: pd.DataFrame, symbol: str) -> pd.DataFrame:
        # 预留更复杂多周期过滤接口
        return d

    def run_symbol_backtest(
        self,
        symbol: str,
        df: pd.DataFrame,
        initial_capital: float,
    ) -> SymbolResult:
        params = self._get_params_for_symbol(symbol)

        d = compute_indicators(df)
        d = compute_strategy_signals(d)
        d = compute_dynamic_weights_and_ensemble(d)
        d = self._build_filters(d, symbol)

        cash = initial_capital
        position = 0  # 0=空仓, 1=多, -1=空
        size = 0.0
        entry_price = 0.0
        sl_price = 0.0
        tp_price = 0.0

        pnl_total = 0.0
        trades = 0
        wins = 0

        equity = initial_capital
        max_equity = initial_capital
        max_dd_pct = 0.0

        loss_streak = 0
        cooldown_left = 0

        for idx, row in d.iterrows():
            price = float(row["close"])
            atr = float(row["atr"]) if not np.isnan(row["atr"]) else 0.0
            trend_strength = float(row.get("trend_strength", 0.0))
            ensemble_raw = float(row.get("ensemble_raw", 0.0))
            ensemble_dir = float(row.get("ensemble_dir", 0.0))

            # ===== 持仓：止损 / 止盈 / Trailing Stop =====
            if position != 0:
                # Trailing Stop
                if atr > 0 and params["trail_atr_mult"] > 0:
                    if position > 0:
                        new_sl = price - params["trail_atr_mult"] * atr
                        sl_price = max(sl_price, new_sl)
                    else:
                        new_sl = price + params["trail_atr_mult"] * atr
                        sl_price = min(sl_price, new_sl)

                exit_flag = False
                if position > 0:
                    if price <= sl_price or price >= tp_price:
                        exit_flag = True
                else:
                    if price >= sl_price or price <= tp_price:
                        exit_flag = True

                if exit_flag:
                    pnl = (price - entry_price) * size * position
                    pnl_total += pnl
                    cash += pnl
                    trades += 1

                    if pnl > 0:
                        wins += 1
                        loss_streak = 0
                    else:
                        loss_streak += 1
                        if loss_streak >= params["max_loss_streak"]:
                            cooldown_left = params["cooldown_bars"]
                            loss_streak = 0
                            logger.info(
                                "🧊 %s 连续亏损触发冷静期: %d bars", symbol, cooldown_left
                            )

                    position = 0
                    size = 0.0
                    entry_price = 0.0
                    sl_price = 0.0
                    tp_price = 0.0

            # ===== 更新权益 & 回撤 =====
            if position != 0:
                equity = cash + (price - entry_price) * size * position
            else:
                equity = cash

            max_equity = max(max_equity, equity)
            if max_equity > 0:
                dd_pct = (equity - max_equity) / max_equity * 100.0
                max_dd_pct = min(max_dd_pct, dd_pct)

            # ===== 空仓：是否尝试开仓 =====
            if position == 0:
                # 冷静期：禁止新开仓
                if cooldown_left > 0:
                    cooldown_left -= 1
                    continue

                # 多因子过滤
                trend_long_ok = bool(row["trend_long_ok"] and row["rsi_long_ok"])
                trend_short_ok = bool(row["trend_short_ok"] and row["rsi_short_ok"])

                # 策略方向
                long_signal = trend_long_ok and (ensemble_dir > 0)
                short_signal = trend_short_ok and (ensemble_dir < 0)

                if not (long_signal or short_signal):
                    continue

                if atr <= 0:
                    continue

                # Entry Score 打分：趋势 + 策略共识
                consensus_strength = min(1.0, abs(ensemble_raw))
                entry_score = 0.5 * trend_strength + 0.5 * consensus_strength
                if entry_score < 0.6:
                    continue

                # 计算 ATR 止损 / 止盈
                if long_signal:
                    sl_candidate = price - params["sl_atr_mult"] * atr
                    tp_candidate = price + params["tp_atr_mult"] * atr
                    sl_dist = price - sl_candidate
                    tp_dist = tp_candidate - price
                else:
                    sl_candidate = price + params["sl_atr_mult"] * atr
                    tp_candidate = price - params["tp_atr_mult"] * atr
                    sl_dist = sl_candidate - price
                    tp_dist = price - tp_candidate

                if sl_dist <= 0 or tp_dist <= 0:
                    continue

                rr = tp_dist / sl_dist
                if rr < params["min_rr"]:
                    continue

                # 动态风险：趋势越强 → 风险略放大；越弱 → 风险缩小
                base_risk = params["risk_per_trade"]
                dyn_risk = base_risk * (0.5 + trend_strength)  # ∈ [0.5x, 1.5x]
                dyn_risk = max(dyn_risk, base_risk * 0.5)
                dyn_risk = min(dyn_risk, base_risk * 1.5)

                risk_amount = cash * dyn_risk
                if risk_amount <= 0:
                    continue

                size = risk_amount / sl_dist
                if size <= 0:
                    continue

                # 建仓
                position = 1 if long_signal else -1
                entry_price = price
                sl_price = sl_candidate
                tp_price = tp_candidate

        return SymbolResult(
            pnl=pnl_total, trades=trades, wins=wins, max_dd_pct=max_dd_pct
        )


# ============================================================
# 7. 多币种回测
# ============================================================
def run_backtest(
    symbols: List[str],
    days: int,
    initial_capital: float,
    seed: Optional[int],
    data_source: str,
) -> Dict[str, SymbolResult]:
    logger.info("🚀 SmartBacktest V8_1 启动")
    logger.info("🪙 币种: %s", symbols)
    logger.info("📅 回测天数: %d", days)
    logger.info("📊 数据源: %s", data_source)

    if seed is not None:
        np.random.seed(seed)

    engine = AdaptiveSignalEngine()
    market = RealMarketData()

    per_capital = initial_capital / len(symbols)

    results: Dict[str, SymbolResult] = {}
    total_pnl = 0.0
    total_trades = 0
    total_wins = 0
    worst_dd_pct = 0.0

    for sym in symbols:
        logger.info("🔍 处理 %s", sym)

        try:
            if data_source == "real":
                df = market.get_recent_klines(sym, "5m", days)
                if df is None or len(df) == 0:
                    logger.warning("⚠️ %s 真实数据为空，使用模拟市场", sym)
                    df = generate_mock_data(sym, days, seed)
                else:
                    print(f"📥 下载真实K线成功: {sym}, {len(df)} 行")
            else:
                df = generate_mock_data(sym, days, seed)
        except Exception as e:
            logger.error("❌ 获取 %s 真实数据失败: %s", sym, e)
            df = generate_mock_data(sym, days, seed)

        res = engine.run_symbol_backtest(sym, df, per_capital)

        results[sym] = res
        total_pnl += res.pnl
        total_trades += res.trades
        total_wins += res.wins
        worst_dd_pct = min(worst_dd_pct, res.max_dd_pct)

    win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0.0

    print("\n========== 📈 SmartBacktest V8_1 报告 ==========")
    print(f"总收益: {total_pnl:.2f}")
    print(f"总交易数: {total_trades}")
    print(f"总胜率: {win_rate:.2f}%")
    print(f"最大回撤: {worst_dd_pct:.2f}%\n")

    print("按币种：")
    for sym, r in results.items():
        print(
            f"- {sym}: pnl={r.pnl:.2f}, trades={r.trades}, "
            f"win_rate={r.win_rate:.2f}%, maxDD={r.max_dd_pct:.2f}%"
        )

    return results


# ============================================================
# 8. main
# ============================================================
def parse_symbols(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="SmartBacktest V8_1")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC/USDT,ETH/USDT",
        help="逗号分隔交易对，如: BTC/USDT,ETH/USDT",
    )
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["real", "mock"],
        default="real",
        help="real=Binance 真实数据, mock=模拟K线",
    )

    args = parser.parse_args()
    symbols = parse_symbols(args.symbols)

    run_backtest(
        symbols=symbols,
        days=args.days,
        initial_capital=args.initial_capital,
        seed=args.seed,
        data_source=args.data_source,
    )


if __name__ == "__main__":
    main()
