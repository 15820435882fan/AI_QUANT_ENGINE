"""
smart_backtest_v16.py
=================================
V16 · Structure Signal Engine

核心升级点：
1）继续沿用 V15 的缠论结构识别（fractals + BiSegment）
2）新增「三笔结构信号」：
    - 上升结构：up → down → up 且高点抬高、低点抬高 → 多头信号
    - 下降结构：down → up → down 且高点降低、低点降低 → 空头信号
    - 趋势延伸：单笔振幅足够大 → 顺势延伸信号
3）结构因子 + MA 趋势合成 final_trend，用于 regime（trend/mixed/range）判定
4）交易逻辑：
    - 仅在有结构信号时尝试开仓
    - 冷静期 + 连续亏损保护
    - 简单止盈止损：-0.7% 止损 / +1.5% 止盈
"""

import argparse
import logging
from typing import Dict, Any, List, Tuple, Set

import numpy as np
import pandas as pd

from local_data_engine import LocalDataEngine
from structure_engine_v15 import analyze_structure, BiSegment

# ================== 日志配置 ==================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ================== 工具函数 ==================

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

    # tanh 压缩，避免极端值
    slope_val = float(np.tanh(slope_val * 30.0))  # [-1, 1]
    return (slope_val + 1.0) / 2.0  # [0, 1]


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
    valid: List[BiSegment] = []

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


def compute_struct_bias(
    bis: List[BiSegment],
    use_last: int = 10,
) -> Tuple[float, float, float]:
    """
    从最近若干笔评估结构方向偏置（0~1）：
    - 返回: (struct_bias, up_ratio, down_ratio)
      struct_bias 越接近 1 越偏多，接近 0 越偏空，中性 ≈ 0.5
    """
    if not bis:
        return 0.5, 0.5, 0.5

    sub = bis[-use_last:]
    dirs = []

    for bi in sub:
        d = 1.0 if bi.direction == "up" else -1.0
        dirs.append(d)

    dirs_arr = np.array(dirs, dtype=float)
    up_ratio = float(np.mean(dirs_arr > 0))
    down_ratio = float(np.mean(dirs_arr < 0))

    # 方向偏置 [-1,1] → [0,1]
    bias_raw = float(np.mean(dirs_arr))  # -1（全空） ~ +1（全多）
    bias_raw = float(np.tanh(bias_raw))  # 平滑
    struct_bias = (bias_raw + 1.0) / 2.0

    return struct_bias, up_ratio, down_ratio


def decide_regime(final_trend: float) -> str:
    """
    根据 final_trend 决定 regime：
    - >=0.6 → trend
    - <=0.4 → range
    - 中间 → mixed
    """
    if final_trend >= 0.6:
        return "trend"
    elif final_trend <= 0.4:
        return "range"
    else:
        return "mixed"


# ================== 结构信号引擎 ==================

def generate_structure_signals(
    df_ltf: pd.DataFrame,
    valid_bis: List[BiSegment],
) -> Tuple[Set[int], Set[int]]:
    """
    基于「三笔结构」生成信号：
    - up → down → up 且高低点抬升 → long 信号
    - down → up → down 且高低点降低 → short 信号
    - 单笔延伸：length_pct 足够大 → 顺势信号
    信号落点：第三笔结束位置 end_index
    """
    long_signals: Set[int] = set()
    short_signals: Set[int] = set()

    if len(valid_bis) < 2:
        return long_signals, short_signals

    # 确保按时间排序
    valid_bis = sorted(valid_bis, key=lambda b: b.start_index)

    for i in range(2, len(valid_bis)):
        b1 = valid_bis[i - 2]
        b2 = valid_bis[i - 1]
        b3 = valid_bis[i]

        # 三笔基本参数
        d1, d2, d3 = b1.direction, b2.direction, b3.direction
        h1, h2, h3 = b1.high, b2.high, b3.high
        l1, l2, l3 = b1.low, b2.low, b3.low

        # ===== 上升结构：up → down → up，高低点抬升 =====
        if d1 == "up" and d2 == "down" and d3 == "up":
            if (h3 > h2) and (h2 >= h1 * 0.99) and (l2 > l1) and (l3 > l2 * 0.995):
                long_signals.add(b3.end_index)

        # ===== 下降结构：down → up → down，高低点降低 =====
        if d1 == "down" and d2 == "up" and d3 == "down":
            if (l3 < l2) and (l2 <= l1 * 1.01) and (h2 < h1) and (h3 < h2 * 1.005):
                short_signals.add(b3.end_index)

        # ===== 趋势延伸：单笔超大振幅，顺势信号 =====
        if b3.length_pct >= 0.02:  # 单笔振幅 ≥ 2%
            if d3 == "up":
                long_signals.add(b3.end_index)
            else:
                short_signals.add(b3.end_index)

    logging.info(
        f"🧩 结构信号统计: long={len(long_signals)}, short={len(short_signals)}"
    )
    return long_signals, short_signals


# ================== 主策略引擎 ==================

class StructureSignalEngineV16:

    def __init__(self, capital: float = 10000.0):
        self.initial_capital = capital
        self.loss_streak_limit = 3
        self.cooldown_bars = 86  # 5m * 86 ≈ 7 小时

        # 止盈止损参数（可调）
        self.stop_loss = 0.007   # 0.7%
        self.take_profit = 0.015 # 1.5%

    # ---- 单币种回测 ----
    def run_symbol(
        self,
        symbol: str,
        df_ltf: pd.DataFrame,
        df_mtf: pd.DataFrame,
        df_htf: pd.DataFrame,
    ) -> Dict[str, Any]:

        # ===== 1. 缠论结构识别 =====
        fractals, bis = analyze_structure(
            df_ltf,
            left=2,
            right=2,
            min_fractal_strength=0.0,
            min_bars=3,
            min_move_pct=0.002,
        )

        valid_bis = filter_valid_bis(bis, min_bars=7, min_move_pct=0.003)

        # ===== 2. 结构方向 + MA 趋势 + Regime =====
        struct_bias, up_ratio, down_ratio = compute_struct_bias(valid_bis, use_last=12)
        ma_score = calc_ma_trend(df_ltf)

        # 综合趋势：结构略偏重
        final_trend = 0.55 * ma_score + 0.45 * struct_bias
        regime = decide_regime(final_trend)

        logging.info(
            f"📐 {symbol} bi_total={len(bis)}, bi_valid={len(valid_bis)}, "
            f"struct_bias={struct_bias:.2f}, up={up_ratio:.2f}, down={down_ratio:.2f}, "
            f"ma={ma_score:.2f}, final={final_trend:.2f}, regime={regime}"
        )

        # ===== 3. 结构信号生成 =====
        long_signals, short_signals = generate_structure_signals(df_ltf, valid_bis)

        closes = df_ltf["close"].values
        highs = df_ltf["high"].values
        lows = df_ltf["low"].values

        position = 0   # 0=空仓，1=多，-1=空
        entry_price = 0.0

        trades: List[float] = []  # 单笔收益（百分比）
        loss_streak = 0
        cooldown = 0

        # ===== 4. 交易主循环 =====
        for i in range(60, len(df_ltf)):  # 留出足够历史计算区间
            price = closes[i]

            # 冷静期
            if cooldown > 0:
                cooldown -= 1
                continue

            # 持仓管理：止盈止损 + 结构反向信号平仓
            if position != 0:
                ret = (price - entry_price) / entry_price * position

                # 止盈止损
                if ret <= -self.stop_loss or ret >= self.take_profit:
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
                    entry_price = 0.0
                    continue

                # 结构反向信号：有明显反向结构信号时平仓
                if position == 1 and i in short_signals:
                    trades.append(ret)
                    loss_streak = loss_streak + 1 if ret < 0 else 0
                    if loss_streak >= self.loss_streak_limit:
                        cooldown = self.cooldown_bars
                        logging.info(
                            f"🧊 {symbol} 多头被反向结构信号止损 → 冷静 {self.cooldown_bars} bars"
                        )
                    position = 0
                    entry_price = 0.0
                    continue

                if position == -1 and i in long_signals:
                    trades.append(ret)
                    loss_streak = loss_streak + 1 if ret < 0 else 0
                    if loss_streak >= self.loss_streak_limit:
                        cooldown = self.cooldown_bars
                        logging.info(
                            f"🧊 {symbol} 空头被反向结构信号止损 → 冷静 {self.cooldown_bars} bars"
                        )
                    position = 0
                    entry_price = 0.0
                    continue

            # 空仓状态 → 根据结构信号 + regime 开仓
            if position == 0:
                # 趋势/混合市场下的多头优先条件
                allow_long = (final_trend >= 0.48)
                # 趋势/混合市场下的空头优先条件
                allow_short = (final_trend <= 0.52)

                # mixed 模式下更保守：要求结构信号 + 一点点价格确认
                if regime == "trend":
                    # 纯趋势：结构信号直接执行
                    if allow_long and i in long_signals:
                        position = 1
                        entry_price = price
                    elif allow_short and i in short_signals:
                        position = -1
                        entry_price = price

                elif regime == "mixed":
                    # 混合模式：参考最近区间高低点
                    window = 40
                    if i > window:
                        recent_high = highs[i - window : i].max()
                        recent_low = lows[i - window : i].min()
                    else:
                        recent_high = highs[:i].max()
                        recent_low = lows[:i].min()

                    if allow_long and i in long_signals and price > recent_high * 0.998:
                        position = 1
                        entry_price = price
                    elif allow_short and i in short_signals and price < recent_low * 1.002:
                        position = -1
                        entry_price = price

                else:  # range
                    # 震荡：只做“结构反转”信号
                    window = 30
                    if i > window:
                        recent_high = highs[i - window : i].max()
                        recent_low = lows[i - window : i].min()
                    else:
                        recent_high = highs[:i].max()
                        recent_low = lows[:i].min()

                    if i in long_signals and price <= recent_low * 1.002:
                        position = 1
                        entry_price = price
                    elif i in short_signals and price >= recent_high * 0.998:
                        position = -1
                        entry_price = price

        # 收盘如有剩余头寸，按最后价格平仓
        if position != 0 and entry_price > 0:
            last_price = closes[-1]
            ret = (last_price - entry_price) / entry_price * position
            trades.append(ret)

        # ===== 5. 汇总结果 =====
        total_ret = sum(trades)
        pnl = total_ret * self.initial_capital
        trade_cnt = len(trades)
        win_cnt = sum(1 for r in trades if r > 0)
        win_rate = win_cnt / trade_cnt * 100 if trade_cnt > 0 else 0.0

        return {
            "symbol": symbol,
            "pnl": pnl,
            "trades": trade_cnt,
            "win_rate": win_rate,
            "regime": regime,
            "struct_bias": struct_bias,
            "ma_score": ma_score,
            "final_trend": final_trend,
            "bi_total": len(bis),
            "bi_valid": len(valid_bis),
            "fractals": len(fractals),
            "struct_long_signals": len(long_signals),
            "struct_short_signals": len(short_signals),
        }


# ================== 主入口 ==================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--data-source", type=str, default="local")
    parser.add_argument("--capital", type=float, default=10000.0)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    logging.info("🚀 SmartBacktest V16 启动")
    logging.info(f"🪙 币种: {symbols}")
    logging.info(f"📅 回测天数: {args.days}")
    logging.info(f"📊 数据源: {args.data_source}")

    if args.data_source != "local":
        logging.warning("⚠️ 当前 V16 建议使用本地数据，请先用 download_all_data.py 下载。")

    engine = LocalDataEngine(base_dir="data", exchange="binance")
    core = StructureSignalEngineV16(capital=args.capital)

    results: Dict[str, Any] = {}

    for sym in symbols:
        logging.info(f"🔍 处理 {sym}")
        df_ltf = engine.load_klines(sym, "5m", args.days)
        df_mtf = engine.load_klines(sym, "1h", args.days + 3)
        df_htf = engine.load_klines(sym, "4h", args.days + 7)

        res = core.run_symbol(sym, df_ltf, df_mtf, df_htf)
        results[sym] = res

    # ===== 汇总输出 =====
    total_pnl = sum(r["pnl"] for r in results.values())
    total_trades = sum(r["trades"] for r in results.values())

    print("\n========== 📈 SmartBacktest V16 报告 ==========")
    print(f"总收益: {total_pnl:.2f}")
    print(f"总交易数: {total_trades}")

    for sym, r in results.items():
        print(
            f"\n- {sym}: pnl={r['pnl']:.2f}, trades={r['trades']}, "
            f"win_rate={r['win_rate']:.2f}%, regime={r['regime']}, "
            f"struct_bias={r['struct_bias']:.2f}, ma={r['ma_score']:.2f}, "
            f"final={r['final_trend']:.2f}, "
            f"Bi_total={r['bi_total']}, Bi_valid={r['bi_valid']}, "
            f"fractals={r['fractals']}, "
            f"long_signals={r['struct_long_signals']}, short_signals={r['struct_short_signals']}"
        )


if __name__ == "__main__":
    main()
