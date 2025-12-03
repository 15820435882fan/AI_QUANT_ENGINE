# v30_mtf_feature_builder.py
#
# 多周期(MTF: Multi Time Frame)特征构造：
#   - 基础周期：1h
#   - 中周期： 15m
#   - 小周期： 5m
#
# 目标：在每一根 1h K 上，附加该小时内部 15m / 5m 的统计特征，
#      例如收益均值、波动率等，让模型对小级别结构有感知。
#
# 使用对象：
#   - V30 强化学习环境（v30_rl_env.py）
#   - 后续如需构建 MTF 版监督数据集也可以复用。
#
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from v30_dataset_builder import load_ohlc, build_features
from v30_teacher_strategies import teacher_trend_follow, TeacherConfig


def build_mtf_features(
    symbol: str,
    base_timeframe: str = "1h",
    mid_timeframe: str = "15m",
    fast_timeframe: str = "5m",
    days: int = 365,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    返回：
        df_mtf: 带有老师动作 + 单周期基础特征 + 15m/5m 统计特征的 DataFrame
        feat_cols: 特征列名列表
    """
    # 1) 加载基础周期 K 线
    df_base_raw = load_ohlc(symbol, base_timeframe, days)
    teacher_cfg = TeacherConfig()
    df_teacher = teacher_trend_follow(df_base_raw, cfg=teacher_cfg)   # 带 action_teacher / risk_level 等
    df_base_feat, base_feat_cols = build_features(df_teacher)         # 构建 1h 自身特征

    # 2) 加载中周期 / 小周期 K 线
    df_mid = load_ohlc(symbol, mid_timeframe, days)
    df_fast = load_ohlc(symbol, fast_timeframe, days)

    # 转换为 pandas 时间索引
    for df in (df_mid, df_fast):
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

    # 计算 15m / 5m 的收益
    df_mid["ret_1"] = df_mid["close"].pct_change()
    df_fast["ret_1"] = df_fast["close"].pct_change()

    # 按 1h 重采样，构造统计特征
    agg_dict = {
        "ret_1": ["mean", "std"],
        "high": ["max"],
        "low": ["min"],
    }

    mid_agg = df_mid.resample("1H").agg(agg_dict)
    fast_agg = df_fast.resample("1H").agg(agg_dict)

    # 展开多层列名
    mid_agg.columns = [f"m15_{c}_{s}" for c, s in mid_agg.columns]
    fast_agg.columns = [f"m5_{c}_{s}" for c, s in fast_agg.columns]

    # 对齐到 base 1h 的时间索引
    mid_agg = mid_agg.reindex(df_base_feat.index)
    fast_agg = fast_agg.reindex(df_base_feat.index)

    # 合并
    df_mtf = pd.concat([df_base_feat, mid_agg, fast_agg], axis=1)

    # 对新增特征做简单缺失填充
    df_mtf = df_mtf.fillna(method="ffill").fillna(method="bfill")

    # 构建特征列名：基础特征 + MTF 特征
    mtf_cols = [c for c in df_mtf.columns if c.startswith("m15_") or c.startswith("m5_")]
    feat_cols = base_feat_cols + mtf_cols

    return df_mtf, feat_cols
