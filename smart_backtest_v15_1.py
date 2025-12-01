"""
smart_backtest_v15_1.py · 集成缠论结构引擎版本（第一阶段）

- 使用 structure_engine_v15.analyze_structure 获取 笔（Bi）
- 计算 Structure Trend Score
- 与 MA Trend Score 融合 → final_trend
- 根据 final_trend 判定 regime = trend / range
- 交易逻辑保持简化版，作为后续结构增强的基准版本
"""

import argparse
import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from local_data_engine import LocalDataEngine
from structure_engine_v15 import analyze_structure, BiSegment

# ========= 日志配置 =========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ========= 工具函数 =========

def calc_ma_trend(df: pd.DataFrame) -> float:
    """
    MA 趋势分数（0~1）
    用 5m 的 MA20 斜率来衡量趋势方向和力度
    """
    if len(df) < 50:
        return 0.5

    close = df["close"]
    ma20 = close.rolling(20).mean()
    slope = (ma20 - ma20.shift(1)) / ma20.shift(1)

    slope_val = slope.iloc[-1]
    slope_val = float(np.tanh(slope_val * 50.0))  # 映射到 [-1,1]

    return (slope_val + 1.0) / 2.0  # 转为 [0,1]


def structure_trend_score(bis: List[BiSegment]) -> float:
    """
    缠论结构趋势得分：根据最后一笔的方向 + 长度 + 斜率 来打分（0~1）
    """
    if not bis:
        return 0.5

    last_bi = bis[-1]

    direction = 1.0 if last_bi.direction == "up" else -1.0
    # length_pct 通常在 0~0.05 这个量级，放大一点做非线性
    raw = direction * (last_bi.length_pct * 20.0 + last_bi.slope * 50.0)
    raw = float(np.tanh(raw))  # 压到 [-1,1]

    return (raw + 1.0) / 2.0


# ========= 主策略引擎 =========

class SmartBacktestV15:

    def __init__(self):
        # 连续亏损熄火
        self.loss_streak_limit = 3
        self.cooldown_bars = 86  # 冷静期 bars 数（5m * 86 ≈ 430min ≈ 7h）

    def run_symbol(
        self,
        symbol: str,
        df_ltf: pd.DataFrame,
        df_mtf: pd.DataFrame,
        df_htf: pd.DataFrame,
        capital: float = 10000.0,
    ) -> Dict[str, Any]:
        """
        对单一 symbol 运行回测（只使用 5m 进行交易决策，
        1h/4h 预留给后续结构增强用）
        """

        # ========== 结构分析 ==========
        fractals, bis = analyze_structure(
            df_ltf,
            left=2,
            right=2,
            min_fractal_strength=0.0,
            min_bars=3,
            min_move_pct=0.002,
        )

        struct_score = structure_trend_score(bis)
        ma_score = calc_ma_trend(df_ltf)
        final_trend = 0.6 * ma_score + 0.4 * struct_score

        regime = "trend" if final_trend >= 0.5 else "range"

        logging.info(
            f"📐 {symbol} structure={struct_score:.2f}, ma={ma_score:.2f}, "
            f"final={final_trend:.2f}, regime={regime}"
        )

        # ========== 交易回测（简版） ==========
        closes = df_ltf["close"].values
        highs = df_ltf["high"].values
        lows = df_ltf["low"].values

        position = 0       # 1=多，-1=空，0=空仓
        entry = 0.0
        trades_ret: List[float] = []
        loss_streak = 0
        cooldown = 0

        # 为了简单，固定单次交易使用 1 单位资金（后面可以接仓位管理）
        for i in range(50, len(df_ltf)):  # 从50开始避免 rolling NaN 太多
            price = closes[i]

            # 冷静期
            if cooldown > 0:
                cooldown -= 1
                continue

            # 持仓管理
            if position != 0:
                # 简单盈亏计算
                ret = (price - entry) / entry * position

                # 简单止盈止损：0.5% 止损 / 1% 止盈
                if ret <= -0.005 or ret >= 0.01:
                    trades_ret.append(ret)

                    if ret < 0:
                        loss_streak += 1
                        if loss_streak >= self.loss_streak_limit:
                            cooldown = self.cooldown_bars
                            logging.info(
                                f"🧊 {symbol} 连续亏损({loss_streak}) → 冷静 {self.cooldown_bars} bars"
                            )
                    else:
                        loss_streak = 0

                    position = 0
                    entry = 0.0
                    continue

            # 空仓 → 考虑开仓
            if position == 0:
                if regime == "trend":
                    # 趋势模式：突破开仓
                    prev_high_max = highs[i-30:i].max()
                    prev_low_min = lows[i-30:i].min()

                    if price > prev_high_max:
                        position = 1
                        entry = price
                    elif price < prev_low_min:
                        position = -1
                        entry = price

                else:
                    # 震荡模式：反向开仓
                    prev_low_min = lows[i-20:i].min()
                    prev_high_max = highs[i-20:i].max()

                    if price < prev_low_min:
                        position = 1
                        entry = price
                    elif price > prev_high_max:
                        position = -1
                        entry = price

        # 统计结果
        total_ret = sum(trades_ret)
        pnl = total_ret * capital

        win_cnt = sum(1 for r in trades_ret if r > 0)
        loss_cnt = sum(1 for r in trades_ret if r < 0)
        trade_cnt = len(trades_ret)
        win_rate = win_cnt / trade_cnt * 100 if trade_cnt > 0 else 0.0

        return {
            "symbol": symbol,
            "pnl": pnl,
            "trades": trade_cnt,
            "win": win_cnt,
            "loss": loss_cnt,
            "win_rate": win_rate,
            "regime": regime,
            "structure_score": struct_score,
            "ma_score": ma_score,
            "final_trend": final_trend,
            "bi_count": len(bis),
            "fractal_count": len(fractals),
        }


# ========= 主入口 =========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--data-source", type=str, default="local")  # 预留 real
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    logging.info("🚀 SmartBacktest V15_1 启动")
    logging.info(f"🪙 币种: {symbols}")
    logging.info(f"📅 回测天数: {args.days}")
    logging.info(f"📊 数据源: {args.data_source}")

    if args.data_source != "local":
        logging.warning("⚠️ 当前 V15_1 暂时只使用本地数据，请先用 download_all_data.py 下载。")

    engine = LocalDataEngine(base_dir="data", exchange="binance")
    core = SmartBacktestV15()

    results: Dict[str, Any] = {}
    for sym in symbols:
        logging.info(f"🔍 处理 {sym}")
        df_ltf = engine.load_klines(sym, "5m", args.days)
        df_mtf = engine.load_klines(sym, "1h", args.days + 3)
        df_htf = engine.load_klines(sym, "4h", args.days + 7)

        res = core.run_symbol(sym, df_ltf, df_mtf, df_htf, capital=10000.0)
        results[sym] = res

    # 汇总
    total_pnl = sum(r["pnl"] for r in results.values())
    total_trades = sum(r["trades"] for r in results.values())
    total_win = sum(r["win"] for r in results.values())
    win_rate = total_win / total_trades * 100 if total_trades > 0 else 0.0

    print("\n========== 📈 SmartBacktest V15_1 报告 ==========")
    print(f"总收益: {total_pnl:.2f}")
    print(f"总交易数: {total_trades}")
    print(f"总胜率: {win_rate:.2f}%")

    for sym, r in results.items():
        print(
            f"\n- {sym}: pnl={r['pnl']:.2f}, trades={r['trades']}, "
            f"win_rate={r['win_rate']:.2f}%, regime={r['regime']}, "
            f"structure={r['structure_score']:.2f}, ma={r['ma_score']:.2f}, "
            f"final={r['final_trend']:.2f}, bi={r['bi_count']}, fractals={r['fractal_count']}"
        )


if __name__ == "__main__":
    main()
