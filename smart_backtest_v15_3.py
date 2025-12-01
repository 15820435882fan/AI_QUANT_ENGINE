"""
smart_backtest_v15_3.py
- 缠论结构因子重新标定版
- 目标：让 struct_score 在 0~1 之间合理波动，而不是长期 ≈ 1.0
- 综合趋势：final_trend = 0.7 * ma_trend + 0.3 * struct_trend
"""

import argparse
import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from local_data_engine import LocalDataEngine
from structure_engine_v15 import analyze_structure, BiSegment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ========= MA 趋势打分 =========

def calc_ma_trend(df: pd.DataFrame) -> float:
    """
    使用 5m close 的 MA20 斜率计算趋势强度（0~1）
    """
    if len(df) < 50:
        return 0.5

    close = df["close"]
    ma20 = close.rolling(20).mean()
    slope = (ma20 - ma20.shift(1)) / ma20.shift(1)

    slope_val = slope.iloc[-1]
    # 轻一点的 tanh，避免过饱和
    slope_val = float(np.tanh(slope_val * 30.0))  # [-1,1]

    return (slope_val + 1.0) / 2.0


# ========= 缠论笔过滤 =========

def filter_valid_bis(
    bis: List[BiSegment],
    min_bars: int = 7,
    min_move_pct: float = 0.003,
) -> List[BiSegment]:
    """
    过滤“有效笔”：
    - 至少 min_bars 根K
    - 涨跌幅 ≥ min_move_pct
    """
    valid = []
    for bi in bis:
        bars = getattr(bi, "bar_count", None)
        if bars is None:
            try:
                bars = int(bi.end_index - bi.start_index + 1)
            except Exception:
                bars = 0

        if bars < min_bars:
            continue

        if bi.length_pct < min_move_pct:
            continue

        valid.append(bi)

    return valid


# ========= 三笔结构趋势打分（重新标定） =========

def structure_trend_score_three_bis(bis: List[BiSegment]) -> float:
    """
    使用最近三笔（不足则用所有）计算结构趋势得分（0~1）：
    - 方向一致性：三笔方向同向越强
    - 振幅 length_pct：和参考值比较做归一化
    - 斜率 slope：和参考值比较做归一化

    设计目标：
    - 大多数时间 struct 在 0.3~0.7 间波动
    - 只有“方向高度一致 + 振幅大 + 斜率大”的情况才接近 0.8~1.0
    """
    if len(bis) == 0:
        return 0.5

    use_n = min(3, len(bis))
    last = bis[-use_n:]

    # 方向：up = +1, down = -1
    dirs = []
    lengths = []
    slopes = []

    for bi in last:
        d = 1.0 if bi.direction == "up" else -1.0
        dirs.append(d)
        lengths.append(bi.length_pct)
        slopes.append(bi.slope)

    dirs = np.array(dirs, dtype=float)
    lengths = np.array(lengths, dtype=float)
    slopes = np.array(slopes, dtype=float)

    # --- 方向一致性得分 ---
    up_ratio = np.mean(dirs > 0)
    down_ratio = np.mean(dirs < 0)
    # 一致性：越偏向单边越接近 1
    direction_consistency = float(abs(up_ratio - down_ratio))  # [0,1]
    # 主方向：up 多则 +1，down 多则 -1，相等则 0
    if up_ratio > down_ratio:
        dir_sign = 1.0
    elif down_ratio > up_ratio:
        dir_sign = -1.0
    else:
        dir_sign = 0.0

    # --- 振幅 & 斜率归一化 ---
    avg_len = float(np.mean(lengths))
    avg_abs_slope = float(np.mean(np.abs(slopes)))

    # 经验参考值：5m 笔振幅 0.5%~3% 较常见
    ref_len = 0.02  # 2%
    norm_len = np.clip(avg_len / ref_len, 0.0, 1.0)

    # 经验参考值：斜率很小，给个保守的基准
    ref_slope = 0.001
    norm_slope = np.clip(avg_abs_slope / ref_slope, 0.0, 1.0)

    # 综合“力度”[0,1]
    strength = 0.6 * norm_len + 0.4 * norm_slope

    # 原始分：方向 * 一致性 * 力度，范围 [-1,1]
    raw = dir_sign * direction_consistency * strength

    # 再加一层轻微 tanh，防波动过大
    raw = float(np.tanh(raw * 2.0))  # still in [-1,1] but平滑

    return (raw + 1.0) / 2.0  # [0,1]


# ========= Regime 判定 =========

def decide_regime(final_trend: float) -> str:
    """
    根据 final_trend 决定 regime
    - >=0.6 视为 trend
    - <=0.4 视为 range
    - 中间 mixed
    """
    if final_trend >= 0.6:
        return "trend"
    elif final_trend <= 0.4:
        return "range"
    else:
        return "mixed"


# ========= 主策略引擎 =========

class SmartBacktestV15_3:

    def __init__(self):
        self.loss_streak_limit = 3
        self.cooldown_bars = 86  # 5m * 86 ≈ 7 小时

    def run_symbol(
        self,
        symbol: str,
        df_ltf: pd.DataFrame,
        df_mtf: pd.DataFrame,
        df_htf: pd.DataFrame,
        capital: float = 10000.0,
    ) -> Dict[str, Any]:
        """
        使用 5m 做交易决策；1h/4h 当前只参与结构分析，不直接下单。
        """

        # ======== 缠论结构识别 ========
        fractals, bis = analyze_structure(
            df_ltf,
            left=2,
            right=2,
            min_fractal_strength=0.0,
            min_bars=3,
            min_move_pct=0.002,
        )

        # 过滤“有效笔”
        valid_bis = filter_valid_bis(bis, min_bars=7, min_move_pct=0.003)

        # 结构趋势评分（三笔）
        struct_score = (
            structure_trend_score_three_bis(valid_bis)
            if valid_bis else 0.5
        )

        # MA 趋势评分
        ma_score = calc_ma_trend(df_ltf)

        # 综合趋势：MA 主导，结构为辅
        final_trend = 0.7 * ma_score + 0.3 * struct_score
        regime = decide_regime(final_trend)

        logging.info(
            f"📐 {symbol} bi_total={len(bis)}, bi_valid={len(valid_bis)}, "
            f"struct={struct_score:.2f}, ma={ma_score:.2f}, "
            f"final={final_trend:.2f}, regime={regime}"
        )

        # ======== 简化交易逻辑（验证结构+regime 有效性） ========
        closes = df_ltf["close"].values
        highs = df_ltf["high"].values
        lows = df_ltf["low"].values

        position = 0   # 1=多，-1=空
        entry = 0.0

        trades: List[float] = []
        loss_streak = 0
        cooldown = 0

        for i in range(50, len(df_ltf)):  # 前面留一段给 rolling 指标
            price = closes[i]

            # 冷静期不交易
            if cooldown > 0:
                cooldown -= 1
                continue

            # 有持仓时 → 止盈止损管理
            if position != 0:
                ret = (price - entry) / entry * position

                # 简单：-0.5% 止损 / +1% 止盈
                if ret <= -0.005 or ret >= 0.01:
                    trades.append(ret)

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

            # 没仓位 → 根据 regime 考虑开仓
            if position == 0:

                if regime == "trend":
                    # 趋势模式：近期高低点突破
                    prev_high = highs[i - 40:i].max()
                    prev_low = lows[i - 40:i].min()

                    if price > prev_high:
                        position = 1
                        entry = price
                    elif price < prev_low:
                        position = -1
                        entry = price

                elif regime == "range":
                    # 震荡模式：在极端位置反向
                    prev_high = highs[i - 25:i].max()
                    prev_low = lows[i - 25:i].min()

                    if price < prev_low * 0.998:
                        position = 1
                        entry = price
                    elif price > prev_high * 1.002:
                        position = -1
                        entry = price

                else:  # mixed
                    # 混合模式：只跟随极端突破
                    prev_high = highs[i - 30:i].max()
                    prev_low = lows[i - 30:i].min()

                    if price > prev_high * 1.002:
                        position = 1
                        entry = price
                    elif price < prev_low * 0.998:
                        position = -1
                        entry = price

        # ======== 统计结果 ========
        total_ret = sum(trades)
        pnl = total_ret * capital

        win_cnt = sum(1 for r in trades if r > 0)
        trade_cnt = len(trades)
        win_rate = win_cnt / trade_cnt * 100 if trade_cnt > 0 else 0.0

        return {
            "symbol": symbol,
            "pnl": pnl,
            "trades": trade_cnt,
            "win_rate": win_rate,
            "regime": regime,
            "structure": struct_score,
            "ma": ma_score,
            "final_trend": final_trend,
            "bi_total": len(bis),
            "bi_valid": len(valid_bis),
            "fractals": len(fractals),
        }


# ========= 主入口 =========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--data-source", type=str, default="local")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    logging.info("🚀 SmartBacktest V15_3 启动")
    logging.info(f"🪙 币种: {symbols}")
    logging.info(f"📅 回测天数: {args.days}")
    logging.info(f"📊 数据源: {args.data_source}")

    if args.data_source != "local":
        logging.warning("⚠️ 当前 V15_3 建议使用本地数据，请先用 download_all_data.py 下载。")

    engine = LocalDataEngine(base_dir="data", exchange="binance")
    core = SmartBacktestV15_3()

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

    print("\n========== 📈 SmartBacktest V15_3 报告 ==========")
    print(f"总收益: {total_pnl:.2f}")
    print(f"总交易数: {total_trades}")

    for sym, r in results.items():
        print(
            f"\n- {sym}: pnl={r['pnl']:.2f}, trades={r['trades']}, "
            f"win_rate={r['win_rate']:.2f}%, regime={r['regime']}, "
            f"struct={r['structure']:.2f}, ma={r['ma']:.2f}, final={r['final_trend']:.2f}, "
            f"Bi_total={r['bi_total']}, Bi_valid={r['bi_valid']}, fractals={r['fractals']}"
        )


if __name__ == "__main__":
    main()
