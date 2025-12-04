# -*- coding: utf-8 -*-
"""
V31_2 · Multi-Asset Trend Ranker · CTA 轮动选币评分模块

功能：
- 对多个币种计算跨周期趋势强度评分 TrendScore
- 使用 1H / 4H HMA + ADX + MACD + 24h 动能
- 输出每根 1H K 下，各个币种趋势评分 & TopK 排名
- 为后续多币轮动交易（组合版 V31）提供“选币信号”

用法示例：
    python multi_trend_ranker.py --symbols BTC,ETH,SOL,APT --days 365 --topk 3 --export scores_v31_2.xlsx
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from local_data_engine_v22_9 import LocalDataEngineV22_9  # 你的本地数据引擎


# ===========================
# 工具函数
# ===========================
def normalize_symbol(sym: str) -> str:
    sym = sym.upper().strip()
    if not sym.endswith("USDT"):
        sym = sym + "USDT"
    return sym


def parse_symbol_list(s: str) -> List[str]:
    parts = [x.strip() for x in s.split(",") if x.strip()]
    return [normalize_symbol(x) for x in parts]


def wma(series: pd.Series, period: int) -> pd.Series:
    if period <= 0:
        return series * np.nan
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(),
        raw=True,
    )


def hma(series: pd.Series, period: int) -> pd.Series:
    if period < 2:
        return series
    half = wma(series, period // 2)
    full = wma(series, period)
    raw = 2 * half - full
    hma_period = int(np.sqrt(period))
    if hma_period < 1:
        hma_period = 1
    return wma(raw, hma_period)


def adx(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    plus_dm = high - prev_high
    minus_dm = prev_low - low

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(period).sum() / (atr + 1e-12))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(period).sum() / (atr + 1e-12))

    dx = (plus_di - minus_di).abs() / ((plus_di + minus_di).abs() + 1e-12) * 100
    adx_val = dx.rolling(period).mean()
    return adx_val


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# ===========================
# 配置
# ===========================
@dataclass
class TrendRankConfig:
    symbols: List[str]
    days: int = 365
    hma_period_1h: int = 20
    hma_period_4h: int = 20
    adx_period_1h: int = 14
    adx_period_4h: int = 14
    adx_strong_th: float = 25.0
    adx_normal_th: float = 20.0

    macd_fast_4h: int = 12
    macd_slow_4h: int = 26
    macd_signal_4h: int = 9

    momentum_lookback_hours: int = 24

    # 权重设定（可以后调参）
    w_trend_dir: float = 0.30
    w_adx: float = 0.40
    w_macd_atr: float = 0.20
    w_momentum: float = 0.10

    topk: int = 2


# ===========================
# 主类：多币种趋势评分
# ===========================
class MultiAssetTrendRanker:
    def __init__(self, cfg: TrendRankConfig):
        self.cfg = cfg
        self.engine = LocalDataEngineV22_9()
        # key: symbol -> DataFrame (1H指标 & TrendScore)
        self.symbol_frames: Dict[str, pd.DataFrame] = {}
        # 合并后的长表
        self.long_scores: pd.DataFrame = pd.DataFrame()
        # TopK 结果
        self.topk_table: pd.DataFrame = pd.DataFrame()

    # -------- 单币数据 & 指标计算（基于1H/4H） --------
    def _load_and_calc_for_symbol(self, symbol: str) -> pd.DataFrame:
        cfg = self.cfg
        # 先加载 1H 数据（如果你本地只有5m，这里可以改成先load 5m再resample到1H）
        df_1h = self.engine.load_klines(symbol, "1h", days=cfg.days)

        # ★ 修复：统一去掉时区（关键补丁）
        if isinstance(df_1h.index, pd.DatetimeIndex):
            if df_1h.index.tz is not None:
                df_1h.index = df_1h.index.tz_convert(None)

        df_1h = df_1h.sort_index()

        # 4H 重采样
        ohlc = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        df_4h = df_1h.resample("4H").agg(ohlc).dropna()
        # ★ 修复：确保4h索引无时区
        if isinstance(df_4h.index, pd.DatetimeIndex):
            if df_4h.index.tz is not None:
                df_4h.index = df_4h.index.tz_convert(None)

        # HMA
        df_1h["hma_1h"] = hma(df_1h["close"], cfg.hma_period_1h)
        df_4h["hma_4h"] = hma(df_4h["close"], cfg.hma_period_4h)

        # HMA 方向：1 = 多头，-1 = 空头
        df_1h["hma_dir_1h"] = np.where(df_1h["close"] > df_1h["hma_1h"], 1, -1)
        df_4h["hma_dir_4h"] = np.where(df_4h["close"] > df_4h["hma_4h"], 1, -1)

        # ADX
        df_1h["adx_1h"] = adx(df_1h, cfg.adx_period_1h)
        df_4h["adx_4h"] = adx(df_4h, cfg.adx_period_4h)

        # 对齐 4H → 1H
        df_4h_re = df_4h[["hma_dir_4h", "adx_4h"]].reindex(df_1h.index, method="ffill")
        df_1h["hma_dir_4h"] = df_4h_re["hma_dir_4h"].fillna(0).astype(int)
        df_1h["adx_4h"] = df_4h_re["adx_4h"].fillna(0.0)

        # 趋势方向一致性
        same_sign = (df_1h["hma_dir_1h"] * df_1h["hma_dir_4h"]) > 0
        trend_dir = np.where(same_sign, df_1h["hma_dir_1h"], 0)  # 1=多, -1=空, 0=无明确趋势
        df_1h["trend_dir"] = trend_dir

        # 趋势强度：综合 ADX
        adx_combined = (df_1h["adx_1h"].fillna(0.0) + df_1h["adx_4h"].fillna(0.0)) / 2.0
        df_1h["adx_combined"] = adx_combined

        trend_strength = np.zeros(len(df_1h), dtype=int)
        strong = adx_combined >= cfg.adx_strong_th
        normal = (adx_combined >= cfg.adx_normal_th) & (adx_combined < cfg.adx_strong_th)
        trend_strength[normal] = 1
        trend_strength[strong] = 2
        trend_strength[trend_dir == 0] = 0
        df_1h["trend_strength"] = trend_strength

        # ATR 作为波动度 & 把 MACD 归一化
        df_1h["atr_1h"] = self._atr(df_1h, 14)

        # 4H MACD（用来表达中期动能）
        df_4h["ema_fast"] = ema(df_4h["close"], cfg.macd_fast_4h)
        df_4h["ema_slow"] = ema(df_4h["close"], cfg.macd_slow_4h)
        df_4h["macd"] = df_4h["ema_fast"] - df_4h["ema_slow"]
        df_4h["macd_signal"] = ema(df_4h["macd"], cfg.macd_signal_4h)
        df_4h["macd_hist"] = df_4h["macd"] - df_4h["macd_signal"]

        df_4h_macd = df_4h[["macd_hist"]].reindex(df_1h.index, method="ffill")
        df_1h["macd_hist_4h"] = df_4h_macd["macd_hist"].fillna(0.0)

        # 24H 动能：log return（避免绝对价格差异）
        close = df_1h["close"]
        lookback = cfg.momentum_lookback_hours
        df_1h["ret_24h"] = np.log(close / close.shift(lookback))

        # 评分的各因子计算
        # 1）趋势方向一致性评分：HMA_1h 与 HMA_4h 一致 → 分值 1，不一致 → 0
        trend_dir_score = np.where(trend_dir != 0, 1.0, 0.0)

        # 2）ADX 评分：把 adx_combined 映射到 0~1（假设0~50之间）
        adx_score = (adx_combined.clip(0, 50) / 50.0).fillna(0.0)

        # 3）MACD/ATR 归一化：动能除以波动，表示“单位波动下的方向性”
        atr_safe = df_1h["atr_1h"].replace(0, np.nan)
        macd_atr_raw = df_1h["macd_hist_4h"] / atr_safe
        macd_atr_raw = macd_atr_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 为了稳定，把 macd_atr 限制在 [-3, 3]，再线性映射到 [0,1]
        macd_atr_clipped = macd_atr_raw.clip(-3, 3)
        macd_atr_score = (macd_atr_clipped + 3) / 6.0  # -3→0, +3→1

        # 4）24h 动能评分：同样剪裁 [-0.1, 0.1] ≈ +/-10%
        mom_raw = df_1h["ret_24h"]
        mom_clipped = mom_raw.clip(-0.1, 0.1)
        momentum_score = (mom_clipped + 0.1) / 0.2  # -0.1→0, +0.1→1
        momentum_score = momentum_score.fillna(0.5)  # 没有数据时给中性 0.5

        # 综合 TrendScore
        cfgw = self.cfg
        trend_score = (
            cfgw.w_trend_dir * trend_dir_score +
            cfgw.w_adx * adx_score +
            cfgw.w_macd_atr * macd_atr_score +
            cfgw.w_momentum * momentum_score
        )

        df_1h["trend_dir_score"] = trend_dir_score
        df_1h["adx_score"] = adx_score
        df_1h["macd_atr_score"] = macd_atr_score
        df_1h["momentum_score"] = momentum_score
        df_1h["TrendScore"] = trend_score

        # 去掉全是NaN的前面一段
        df_1h = df_1h.dropna(subset=["TrendScore"])

        # 记录币种名
        df_1h["symbol"] = symbol

        return df_1h

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_val = tr.rolling(period).mean()
        return atr_val

    # -------- 主流程：计算所有币种的评分 & TopK --------
    def run(self):
        frames = []
        for sym in self.cfg.symbols:
            print(f"[Ranker] 加载并计算 {sym} ...")
            df_sym = self._load_and_calc_for_symbol(sym)
            self.symbol_frames[sym] = df_sym
            frames.append(df_sym[["symbol", "TrendScore"]])

        if not frames:
            raise ValueError("没有任何币种数据。")

        # 合并所有币种的 TrendScore（长表模式）
        all_scores = pd.concat(frames, axis=0)
        # 确保索引为 DatetimeIndex（去 tz）
        if isinstance(all_scores.index, pd.DatetimeIndex) and all_scores.index.tz is not None:
            all_scores.index = all_scores.index.tz_convert(None)

        all_scores = all_scores.sort_index()
        self.long_scores = all_scores

        # 透视表：index=time, columns=symbol, value=TrendScore
        pivot = all_scores.pivot_table(
            index=all_scores.index,
            columns="symbol",
            values="TrendScore",
            aggfunc="mean",
        ).sort_index()

        # 对每一行做排名 & TopK 标记
        topk = self.cfg.topk
        top_list = []
        for ts, row in pivot.iterrows():
            # row: symbol -> TrendScore
            row_valid = row.dropna()
            if row_valid.empty:
                top_syms = []
            else:
                ranked = row_valid.sort_values(ascending=False)
                top_syms = list(ranked.index[:topk])
            top_entry = {
                "timestamp": ts,
                "TopSymbols": ",".join(top_syms),
            }
            # 也可以记录 Top1, Top2 ...
            for i in range(topk):
                key = f"Top{i+1}"
                top_entry[key] = top_syms[i] if i < len(top_syms) else ""
            top_list.append(top_entry)

        topk_df = pd.DataFrame(top_list).set_index("timestamp")
        self.topk_table = topk_df

        print("\n[Ranker] 趋势评分 & TopK 排名计算完成。")
        self._print_topk_summary(pivot, topk_df)

        return pivot, topk_df

    # -------- TopK 占比统计 --------
    def _print_topk_summary(self, pivot: pd.DataFrame, topk_df: pd.DataFrame):
        print("\n==============================")
        print("多币种 CTA 轮动 · TopK 占比统计")
        print("==============================")

        # 每个 symbol：进入 TopK 的次数/占比
        counts = {sym: 0 for sym in pivot.columns}
        total_rows = len(topk_df)
        for _, row in topk_df.iterrows():
            syms = str(row["TopSymbols"]).split(",") if isinstance(row["TopSymbols"], str) else []
            for s in syms:
                s = s.strip()
                if s in counts:
                    counts[s] += 1

        stats_rows = []
        for sym, c in counts.items():
            ratio = c / total_rows * 100 if total_rows > 0 else 0.0
            stats_rows.append({"symbol": sym, "topk_count": c, "topk_ratio(%)": round(ratio, 2)})
        stats_df = pd.DataFrame(stats_rows).sort_values("topk_ratio(%)", ascending=False)

        print(stats_df.to_string(index=False))

    # -------- 导出 Excel --------
    def export(self, filepath: str):
        if self.long_scores.empty or self.topk_table.empty:
            print("没有可导出的数据，请先运行 run()。")
            return

        output_dir = os.path.dirname(os.path.abspath(filepath))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 透视为宽表：每根1H K，各个币的 TrendScore
        pivot = self.long_scores.pivot_table(
            index=self.long_scores.index,
            columns="symbol",
            values="TrendScore",
            aggfunc="mean",
        ).sort_index()

        # 处理时间列
        pivot_df = pivot.copy()
        pivot_df.insert(0, "timestamp", pivot_df.index)
        if isinstance(pivot_df["timestamp"], pd.Series) and pd.api.types.is_datetime64_any_dtype(pivot_df["timestamp"]):
            tz = getattr(pivot_df["timestamp"].dt, "tz", None)
            if tz is not None:
                pivot_df["timestamp"] = pivot_df["timestamp"].dt.tz_convert(None)

        topk_df = self.topk_table.copy()
        topk_df.insert(0, "timestamp", topk_df.index)
        if isinstance(topk_df["timestamp"], pd.Series) and pd.api.types.is_datetime64_any_dtype(topk_df["timestamp"]):
            tz2 = getattr(topk_df["timestamp"].dt, "tz", None)
            if tz2 is not None:
                topk_df["timestamp"] = topk_df["timestamp"].dt.tz_convert(None)

        summary_rows = []
        # 统计 TopK 占比
        counts = {col: 0 for col in pivot.columns}
        total_rows = len(self.topk_table)
        for _, row in self.topk_table.iterrows():
            syms = str(row["TopSymbols"]).split(",") if isinstance(row["TopSymbols"], str) else []
            for s in syms:
                s = s.strip()
                if s in counts:
                    counts[s] += 1

        for sym, c in counts.items():
            ratio = c / total_rows * 100 if total_rows > 0 else 0.0
            summary_rows.append({"symbol": sym, "topk_count": c, "topk_ratio(%)": round(ratio, 2)})

        stats_df = pd.DataFrame(summary_rows).sort_values("topk_ratio(%)", ascending=False)

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            pivot_df.to_excel(writer, sheet_name="TrendScore_1H", index=False)
            topk_df.to_excel(writer, sheet_name="TopK_Ranking", index=False)
            stats_df.to_excel(writer, sheet_name="TopK_Summary", index=False)

        print(f"[Ranker] 趋势评分和TopK结果已导出: {os.path.abspath(filepath)}")


# ===========================
# 命令行入口
# ===========================
def parse_args():
    p = argparse.ArgumentParser(description="V31_2 Multi-Asset Trend Ranker · CTA轮动选币评分")
    p.add_argument(
        "--symbols",
        type=str,
        default="BTC,ETH,SOL,APT",
        help="要评估的币种列表，例如: BTC,ETH,SOL,APT,DOGE",
    )
    p.add_argument("--days", type=int, default=365, help="回测天数")
    p.add_argument("--topk", type=int, default=2, help="每个时刻选出前K个最强趋势币")
    p.add_argument(
        "--export",
        type=str,
        default="",
        help="导出Excel路径，例如 results_trend_ranker.xlsx",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    symbols = parse_symbol_list(args.symbols)

    print("=" * 80)
    print("V31_2 · 多币种 CTA 趋势评分 / 轮动选币模块")
    print("监控币种:", symbols)
    print(f"回测天数: {args.days}, TopK={args.topk}")
    print("=" * 80)

    cfg = TrendRankConfig(
        symbols=symbols,
        days=args.days,
        topk=args.topk,
    )

    ranker = MultiAssetTrendRanker(cfg)
    pivot, topk_df = ranker.run()

    if args.export:
        ranker.export(args.export)
