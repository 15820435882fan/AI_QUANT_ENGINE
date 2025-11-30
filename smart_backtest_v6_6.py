import numpy as np
import pandas as pd
import logging
import argparse

logger = logging.getLogger("SmartBacktest")


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - SmartBacktest - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def generate_mock_data(symbol: str, days: int, seed: int) -> pd.DataFrame:
    """
    生成更“像样”的模拟 5m K 线数据：
    - 几段不同趋势（上涨 / 下跌 / 震荡）的随机游走
    - 用 log-price 做几何布朗运动风格
    """
    rng = np.random.default_rng(seed)
    n_bars = days * 24 * 12  # 5 分钟一根

    dt = 1 / (24 * 12 * 365)  # 年化时间步长

    # regime 切换：一段一段地给不同的 drift / vol
    regimes = []
    remaining = n_bars
    while remaining > 0:
        length = int(rng.integers(200, 800))
        length = min(length, remaining)
        bias = float(rng.choice([-0.0008, -0.0003, 0.0, 0.0003, 0.0008]))
        vol = float(rng.choice([0.015, 0.02, 0.03]))
        regimes.append((length, bias, vol))
        remaining -= length

    log_prices = [np.log(100.0)]
    for length, bias, vol in regimes:
        for _ in range(length):
            if len(log_prices) >= n_bars:
                break
            eps = rng.normal()
            dlog = bias * dt + vol * np.sqrt(dt) * eps
            log_prices.append(log_prices[-1] + dlog)
        if len(log_prices) >= n_bars:
            break

    log_prices = np.array(log_prices[:n_bars])
    close = np.exp(log_prices)

    # 构造 OHLC
    noise = rng.normal(scale=0.001, size=n_bars)
    open_ = close * (1 + noise)
    high = np.maximum(open_, close) * (1 + np.abs(noise) * 2)
    low = np.minimum(open_, close) * (1 - np.abs(noise) * 2)
    volume = rng.lognormal(mean=10, sigma=0.5, size=n_bars)

    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=n_bars, freq="5T")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    logger.info("📊 使用模拟市场数据: %s (%d 行)", symbol, len(df))
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # EMA 作为趋势过滤
    df["ema_fast"] = close.ewm(span=20, adjust=False).mean()
    df["ema_slow"] = close.ewm(span=60, adjust=False).mean()

    # ATR 百分比，用来控制波动和 SL/TP 距离
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr"] / close

    # RSI14，用来做动量/超买超卖过滤
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    return df


class SmartBacktest:
    def __init__(self, symbols, days, engine="ai_prod", seed=42):
        self.symbols = symbols
        self.days = days
        self.engine = engine
        self.seed = seed
        self.initial_capital = 10_000.0
        self.leverage = 3.0

    def run_symbol(self, symbol: str, seed_offset: int):
        df = generate_mock_data(symbol, self.days, self.seed + seed_offset)
        df = compute_indicators(df)

        # 简单起见：每个币种独立 1/N 资金池
        balance = self.initial_capital / len(self.symbols)
        equity = balance
        peak_equity = balance
        max_dd = 0.0  # 负数，最后再转为百分比

        position_qty = 0.0
        entry_price = 0.0
        direction = 0  # +1 = 多，-1 = 空
        stop_price = 0.0
        take_price = 0.0
        bars_in_pos = 0

        trades = []

        # 参数（可以以后暴露成 CLI 或配置）
        trend_thr = 0.0025
        min_atr_pct = 0.003
        max_atr_pct = 0.05
        max_hold_bars = 48  # 最多持仓 4 小时（5m bar）

        start = 60  # 指标暖机

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        ema_fast = df["ema_fast"].values
        ema_slow = df["ema_slow"].values
        atr_pct = df["atr_pct"].values
        rsi = df["rsi"].values

        for i in range(start, len(df)):
            price = float(close[i])
            vol_p = float(atr_pct[i])

            if (
                np.isnan(ema_fast[i])
                or np.isnan(ema_slow[i])
                or np.isnan(vol_p)
                or np.isnan(rsi[i])
            ):
                continue

            # === 管理已有仓位 ===
            if direction != 0:
                bars_in_pos += 1
                hit_sl = False
                hit_tp = False

                if direction == 1:
                    # 多头：先看 SL 再看 TP（保守假设）
                    if low[i] <= stop_price:
                        hit_sl = True
                    elif high[i] >= take_price:
                        hit_tp = True
                else:
                    # 空头：先看 SL 再看 TP
                    if high[i] >= stop_price:
                        hit_sl = True
                    elif low[i] <= take_price:
                        hit_tp = True

                exit = False
                exit_price = price
                reason = ""

                if hit_sl:
                    exit = True
                    exit_price = stop_price
                    reason = "SL"
                elif hit_tp:
                    exit = True
                    exit_price = take_price
                    reason = "TP"
                elif bars_in_pos >= max_hold_bars:
                    exit = True
                    reason = "timeout"

                if exit:
                    pnl = (exit_price - entry_price) * position_qty
                    balance += pnl
                    equity = balance
                    peak_equity = max(peak_equity, equity)
                    if peak_equity > 0:
                        dd = (equity - peak_equity) / peak_equity
                        max_dd = min(max_dd, dd)

                    trades.append(
                        {
                            "pnl": pnl,
                            "direction": direction,
                            "entry": entry_price,
                            "exit": exit_price,
                            "reason": reason,
                        }
                    )

                    # 平仓
                    position_qty = 0.0
                    direction = 0
                    bars_in_pos = 0

            # === 空仓时再找信号 ===
            if direction == 0:
                # 波动过滤：过低&过高波动都不做
                if not (min_atr_pct <= vol_p <= max_atr_pct):
                    continue

                ema_diff = (ema_fast[i] - ema_slow[i]) / ema_slow[i]

                sig_dir = 0
                # 多头：有一定向上趋势，RSI>55
                if ema_diff > trend_thr and rsi[i] > 55:
                    sig_dir = 1
                # 空头：向下趋势，RSI<45
                elif ema_diff < -trend_thr and rsi[i] < 45:
                    sig_dir = -1

                if sig_dir == 0:
                    continue

                # === 动态仓位：跟随趋势强度 & 波动 ===
                base_risk = 0.01  # 单笔基础风险 1%
                trend_strength = min(2.0, abs(ema_diff) / trend_thr)
                risk_frac = base_risk * (0.5 + 0.5 * trend_strength)

                # 波动太大/太小时缩仓
                if vol_p > 0.03:
                    risk_frac *= 0.7
                elif vol_p < 0.006:
                    risk_frac *= 0.5

                risk_frac = float(np.clip(risk_frac, 0.003, 0.03))

                max_notional = balance * self.leverage
                trade_notional = max_notional * risk_frac
                if trade_notional < 10:  # 太小不做
                    continue

                qty = (trade_notional / price) * sig_dir

                # SL/TP 由 ATR 百分比决定（自适应市场波动）
                risk_sl = float(np.clip(1.5 * vol_p, 0.003, 0.02))
                tp_mult = float(np.clip(2.5 * vol_p, 0.006, 0.04))
                if sig_dir == 1:
                    stop = price * (1 - risk_sl)
                    take = price * (1 + tp_mult)
                else:
                    stop = price * (1 + risk_sl)
                    take = price * (1 - tp_mult)

                position_qty = qty
                entry_price = price
                direction = sig_dir
                stop_price = stop
                take_price = take
                bars_in_pos = 0

        # 回测结束如还有仓位，按最后一根 K 线平掉
        if direction != 0:
            final_price = float(close[-1])
            pnl = (final_price - entry_price) * position_qty
            balance += pnl
            trades.append(
                {
                    "pnl": pnl,
                    "direction": direction,
                    "entry": entry_price,
                    "exit": final_price,
                    "reason": "eod",
                }
            )

        n_trades = len(trades)
        wins = sum(1 for t in trades if t["pnl"] > 0)
        win_rate = wins / n_trades * 100 if n_trades > 0 else 0.0
        total_pnl = sum(t["pnl"] for t in trades)
        max_dd_pct = -max_dd * 100  # 转为正的百分比

        return {
            "symbol": symbol,
            "final_balance": balance,
            "trades": n_trades,
            "win_rate": win_rate,
            "pnl": total_pnl,
            "max_dd_pct": max_dd_pct,
        }

    def run(self):
        results = []
        for idx, sym in enumerate(self.symbols):
            logger.info("🔍 测试币种: %s", sym)
            stats = self.run_symbol(sym, idx * 1000)
            results.append(stats)

        total_final = sum(r["final_balance"] for r in results)
        total_pnl = total_final - self.initial_capital
        total_trades = sum(r["trades"] for r in results)
        avg_win_rate = (
            sum(r["win_rate"] * r["trades"] for r in results) / total_trades
            if total_trades > 0
            else 0.0
        )
        worst_dd = max((r["max_dd_pct"] for r in results), default=0.0)
        total_return_pct = (total_final / self.initial_capital - 1) * 100

        # 一个简单的“AI 打分”占位：收益 vs 回撤
        score = 50 + total_return_pct * 0.3 - worst_dd * 0.7
        score = float(np.clip(score, 0, 100))

        logger.info("=" * 80)
        logger.info("🧠 智能量化交易系统 - 回测报告")
        logger.info("=" * 80)
        logger.info("测试币种: %d 个", len(self.symbols))
        logger.info("总交易次数: %d 笔", total_trades)
        logger.info("总收益: $%.2f (%.2f%%)", total_pnl, total_return_pct)
        logger.info("最终资金: $%.2f", total_final)
        logger.info("平均胜率: %.1f%%", avg_win_rate)
        logger.info("最大回撤(最差单币种): %.1f%%", worst_dd)
        logger.info("")
        logger.info("📊 各币种表现:")
        for r in results:
            logger.info(
                "  🟡 %s: %d 笔, 胜率: %.1f%%, 收益: $%.2f, 最大回撤: %.1f%%",
                r["symbol"],
                r["trades"],
                r["win_rate"],
                r["pnl"],
                r["max_dd_pct"],
            )
        logger.info("")
        logger.info("🤖 简易风险收益评分: %.1f / 100", score)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--symbols",
        type=str,
        default="BTC/USDT,ETH/USDT,SOL/USDT",
        help="用逗号分隔的交易对列表，例如 BTC/USDT,ETH/USDT",
    )
    p.add_argument("--days", type=int, default=30, help="回测天数")
    p.add_argument(
        "--engine",
        type=str,
        default="ai_prod",
        choices=["ai_prod", "baseline"],
        help="策略引擎（目前 ai_prod/baseline 只是占位，逻辑相同）",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（保证模拟数据与结果可复现）",
    )
    return p.parse_args()


def main():
    setup_logger()
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    logger.info("🚀 开始回测 ...")
    bt = SmartBacktest(
        symbols=symbols,
        days=args.days,
        engine=args.engine,
        seed=args.seed,
    )
    bt.run()


if __name__ == "__main__":
    main()
