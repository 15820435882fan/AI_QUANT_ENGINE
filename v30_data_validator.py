# v30_data_validator.py
#
# V30 数据完整性验证工具
#
# 用途：
#   在训练 / 回测 / 强化学习之前，先检查本地 K 线数据是否完整、是否连续。
#
# 使用示例（在项目根目录运行）：
#
#   python v30_data_validator.py --symbol BTCUSDT --timeframe 1h --days 2920
#   python v30_data_validator.py --symbol BTCUSDT --timeframe 15m --days 1095
#
# 会输出：
#   - 实际数据起止时间
#   - 实际包含的天数 / 根数
#   - 理论根数 vs 实际根数
#   - 缺口（missing gap）统计
#
from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
import pandas as pd

from v30_dataset_builder import load_ohlc


def infer_expected_seconds(timeframe: str) -> int:
    tf = timeframe.lower()
    if tf.endswith("m"):
        minutes = int(tf[:-1])
        return minutes * 60
    if tf.endswith("h"):
        hours = int(tf[:-1])
        return hours * 3600
    if tf.endswith("d"):
        days = int(tf[:-1])
        return days * 86400
    raise ValueError(f"无法识别的 timeframe: {timeframe}")


def analyze_gaps(df: pd.DataFrame, timeframe: str) -> Tuple[int, float]:
    if len(df) < 2:
        return 0, 0.0
    idx = df.index.to_series().sort_values()
    diffs = idx.diff().dt.total_seconds().fillna(0)
    expected = infer_expected_seconds(timeframe)
    # 如果间隔 > 1.5 * expected，则认为是缺口
    mask_gap = diffs > (1.5 * expected)
    num_gaps = int(mask_gap.sum())
    if num_gaps == 0:
        return 0, 0.0
    # 用 gap_size / expected 估算缺失的 K 数
    gap_sizes = diffs[mask_gap] / expected
    est_missing_bars = float(gap_sizes.sum())
    return num_gaps, est_missing_bars


def validate_data(symbol: str, timeframe: str, days: int):
    print(f"[DataValidator] 加载数据: symbol={symbol}, timeframe={timeframe}, days={days}")
    df = load_ohlc(symbol, timeframe, days)
    if df.empty:
        print("[DataValidator] ❌ 没有加载到任何数据，请检查本地数据源。")
        return

    df = df.sort_index()
    start = df.index[0]
    end = df.index[-1]
    num_bars = len(df)

    duration_days = (end - start).total_seconds() / 86400.0

    expected_sec = infer_expected_seconds(timeframe)
    theoretical_bars = (end - start).total_seconds() / expected_sec

    num_gaps, est_missing_bars = analyze_gaps(df, timeframe)

    print("========== 数据完整性报告 ==========")
    print(f"交易对: {symbol}, 周期: {timeframe}")
    print(f"请求天数: {days}")
    print(f"实际时间范围: {start}  →  {end}")
    print(f"实际跨度约: {duration_days:.1f} 天")
    print(f"实际 K 线根数: {num_bars}")
    print(f"理论 K 线根数(连续无缺失估算): {theoretical_bars:.1f}")
    print(f"实际 / 理论 比例: {num_bars / max(theoretical_bars, 1):.3f}")
    print(f"检测到缺口数量: {num_gaps}")
    print(f"估算缺失 K 线总数: {est_missing_bars:.1f}")
    if num_gaps == 0 and num_bars / max(theoretical_bars, 1) > 0.95:
        print("结论: ✅ 数据基本连续且完整，可用于长期回测 / 训练。")
    else:
        print("结论: ⚠ 数据存在一定缺失或不完全连续，建议谨慎使用，或缩短时间窗口 / 清洗数据。")
    print("====================================")


def parse_args():
    p = argparse.ArgumentParser(description="V30 数据完整性验证工具")
    p.add_argument("--symbol", type=str, default="BTCUSDT", help="交易对，例如 BTCUSDT")
    p.add_argument("--timeframe", type=str, default="1h", help="周期，例如 1h, 15m")
    p.add_argument("--days", type=int, default=365, help="向前回看多少天的数据")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_data(
        symbol=args.symbol.upper(),
        timeframe=args.timeframe,
        days=args.days,
    )
