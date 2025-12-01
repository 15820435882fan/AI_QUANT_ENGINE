"""
structure_engine_v15.py

V15 第一部分：缠论结构引擎基础模块
- FractalDetector: 分型识别（顶/底分型 + 强度）
- BiDetector:      笔识别（连接分型构成波段）

后续版本将在此基础上加入：
- 中枢识别
- 背驰/盘整结构
- 结构交易信号
"""

from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple

import numpy as np
import pandas as pd

FractalType = Literal["top", "bottom"]


# ===================== 数据结构定义 =====================

@dataclass
class FractalPoint:
    index: int                  # 在原始 df 中的整数索引（基于位置）
    timestamp: pd.Timestamp     # 时间戳
    price: float                # 顶/底对应的 high 或 low
    kind: FractalType           # "top" / "bottom"
    strength: float             # 0~1, 分型强度（相对邻居的突出程度）


@dataclass
class BiSegment:
    start_index: int
    end_index: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    start_price: float
    end_price: float
    direction: Literal["up", "down"]
    length: float               # 绝对涨跌幅
    length_pct: float           # 相对涨跌幅（相对于 start_price）
    bars: int                   # 包含的 K 线数量
    slope: float                # 单位 bar 的平均涨跌
    max_high: float             # 笔内部最高价
    min_low: float              # 笔内部最低价


# ===================== 分型识别器 =====================

class FractalDetector:
    """
    分型识别器（机器版缠论分型）：
    - 顶分型：局部 high 高于左右若干条 K 线的 high
    - 底分型：局部 low 低于左右若干条 K 线的 low
    """

    def __init__(self, left: int = 2, right: int = 2, min_strength: float = 0.0):
        """
        :param left:  左侧参与比较的 K 线数量
        :param right: 右侧参与比较的 K 线数量
        :param min_strength: 分型强度的最小阈值（0~1），太弱的分型可以过滤掉
        """
        self.left = left
        self.right = right
        self.min_strength = min_strength

    def detect(self, df: pd.DataFrame) -> List[FractalPoint]:
        """
        在给定 OHLCV DataFrame 上识别分型。
        需要 df 包含列：["open", "high", "low", "close"]，
        index 为 DatetimeIndex 或有 "timestamp" 列。
        """
        if "high" not in df.columns or "low" not in df.columns:
            raise ValueError("DataFrame 必须包含 'high' 和 'low' 列")

        highs = df["high"].values
        lows = df["low"].values

        # 处理时间戳
        if isinstance(df.index, pd.DatetimeIndex):
            times = df.index
        elif "timestamp" in df.columns:
            times = pd.to_datetime(df["timestamp"])
        else:
            times = pd.to_datetime(df.index)

        n = len(df)
        result: List[FractalPoint] = []

        for i in range(self.left, n - self.right):
            left_slice = slice(i - self.left, i)
            right_slice = slice(i + 1, i + 1 + self.right)

            hi_center = highs[i]
            hi_left = highs[left_slice]
            hi_right = highs[right_slice]

            lo_center = lows[i]
            lo_left = lows[left_slice]
            lo_right = lows[right_slice]

            # ----- 顶分型 -----
            if hi_center > hi_left.max() and hi_center > hi_right.max():
                # 强度：相对左右平均高出多少
                neighbor_hi_mean = (hi_left.mean() + hi_right.mean()) / 2
                strength = (hi_center - neighbor_hi_mean) / max(neighbor_hi_mean, 1e-9)
                strength = float(np.clip(strength, 0.0, 1.0))
                if strength >= self.min_strength:
                    result.append(
                        FractalPoint(
                            index=i,
                            timestamp=times[i],
                            price=float(hi_center),
                            kind="top",
                            strength=strength,
                        )
                    )

            # ----- 底分型 -----
            if lo_center < lo_left.min() and lo_center < lo_right.min():
                neighbor_lo_mean = (lo_left.mean() + lo_right.mean()) / 2
                # 越低 → 强度越大
                strength = (neighbor_lo_mean - lo_center) / max(neighbor_lo_mean, 1e-9)
                strength = float(np.clip(strength, 0.0, 1.0))
                if strength >= self.min_strength:
                    result.append(
                        FractalPoint(
                            index=i,
                            timestamp=times[i],
                            price=float(lo_center),
                            kind="bottom",
                            strength=strength,
                        )
                    )

        # 按 index 排序
        result.sort(key=lambda x: x.index)
        return result


# ===================== 笔识别器 =====================

class BiDetector:
    """
    笔识别器：
    - 输入分型列表
    - 清理相邻“同向分型”（比如连着两个顶分型，只保留更极端那个）
    - 构造相邻顶/底分型之间的“笔”
    """

    def __init__(
        self,
        min_bars: int = 3,
        min_move_pct: float = 0.002,   # 最小波动幅度（0.2%）
    ):
        """
        :param min_bars:      笔至少跨越的 K 线数量，避免极短的噪音
        :param min_move_pct:  笔最小相对涨跌幅阈值（例如 0.002 = 0.2%）
        """
        self.min_bars = min_bars
        self.min_move_pct = min_move_pct

    @staticmethod
    def _deduplicate_fractals(fractals: List[FractalPoint]) -> List[FractalPoint]:
        """
        删除相邻同类型的分型，只保留更极端的：
        - 顶分型：保留更高的
        - 底分型：保留更低的
        """
        if not fractals:
            return []

        cleaned: List[FractalPoint] = [fractals[0]]

        for f in fractals[1:]:
            last = cleaned[-1]
            if f.kind != last.kind:
                cleaned.append(f)
            else:
                # 同类型 → 比较 price 极端程度
                if f.kind == "top":
                    # 价格更高者胜出
                    if f.price > last.price:
                        cleaned[-1] = f
                else:  # bottom
                    if f.price < last.price:
                        cleaned[-1] = f

        return cleaned

    def detect(
        self,
        df: pd.DataFrame,
        fractals: List[FractalPoint],
    ) -> List[BiSegment]:
        """
        根据分型列表识别“笔”（Bi）结构。
        """
        if len(fractals) < 2:
            return []

        # 清理相邻同向分型
        frs = self._deduplicate_fractals(fractals)

        # 再次检查长度
        if len(frs) < 2:
            return []

        if isinstance(df.index, pd.DatetimeIndex):
            times = df.index
        elif "timestamp" in df.columns:
            times = pd.to_datetime(df["timestamp"])
        else:
            times = pd.to_datetime(df.index)

        highs = df["high"].values
        lows = df["low"].values

        bis: List[BiSegment] = []

        # 相邻两个不同类型分型之间形成一笔
        for a, b in zip(frs[:-1], frs[1:]):
            if a.kind == b.kind:
                # 理论上不应该出现，因为 _deduplicate_fractals 已处理
                continue

            start_idx = a.index
            end_idx = b.index
            if end_idx <= start_idx:
                continue

            bars = end_idx - start_idx + 1
            if bars < self.min_bars:
                continue

            start_price = a.price
            end_price = b.price
            move = end_price - start_price
            direction = "up" if move > 0 else "down"
            length = abs(move)
            length_pct = length / max(abs(start_price), 1e-9)

            if length_pct < self.min_move_pct:
                # 波动太小 → 当作噪音，跳过
                continue

            segment_slice = slice(start_idx, end_idx + 1)
            max_high = float(highs[segment_slice].max())
            min_low = float(lows[segment_slice].min())

            slope = move / bars

            bis.append(
                BiSegment(
                    start_index=start_idx,
                    end_index=end_idx,
                    start_time=times[start_idx],
                    end_time=times[end_idx],
                    start_price=float(start_price),
                    end_price=float(end_price),
                    direction=direction,
                    length=float(length),
                    length_pct=float(length_pct),
                    bars=int(bars),
                    slope=float(slope),
                    max_high=max_high,
                    min_low=min_low,
                )
            )

        return bis


# ===================== 一站式结构分析（便于调试） =====================

def analyze_structure(
    df: pd.DataFrame,
    left: int = 2,
    right: int = 2,
    min_fractal_strength: float = 0.0,
    min_bars: int = 3,
    min_move_pct: float = 0.002,
) -> Tuple[List[FractalPoint], List[BiSegment]]:
    """
    一站式：给你 df（单一周期，如 5m），直接返回分型列表 + 笔列表。

    用法示例：
    >>> df = local_engine.load_klines("BTC/USDT", "5m", 60)
    >>> fractals, bis = analyze_structure(df)
    """
    f_detector = FractalDetector(
        left=left,
        right=right,
        min_strength=min_fractal_strength,
    )
    fractals = f_detector.detect(df)

    b_detector = BiDetector(
        min_bars=min_bars,
        min_move_pct=min_move_pct,
    )
    bis = b_detector.detect(df, fractals)

    return fractals, bis


# ===================== 简单自测入口 =====================

if __name__ == "__main__":
    # 这里做一个简单示例：随机生成 K 线（你也可以替换为实际 df 做本地测试）
    import numpy as np

    np.random.seed(42)
    n = 300
    base = np.cumsum(np.random.randn(n)) + 100
    high = base + np.random.rand(n) * 2
    low = base - np.random.rand(n) * 2
    open_ = base + np.random.randn(n) * 0.5
    close = base + np.random.randn(n) * 0.5

    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    df_demo = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.rand(n) * 1000,
        },
        index=idx,
    )

    frs, bis = analyze_structure(df_demo)

    print(f"检测到分型数量: {len(frs)}")
    print(f"检测到笔数量: {len(bis)}")

    if bis:
        print("前 3 笔信息示例：")
        for b in bis[:3]:
            print(
                f"{b.direction} | {b.start_time} -> {b.end_time} | "
                f"{b.start_price:.2f} -> {b.end_price:.2f} | "
                f"bars={b.bars}, pct={b.length_pct*100:.2f}%"
            )
