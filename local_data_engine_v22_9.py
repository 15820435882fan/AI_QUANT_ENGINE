# local_data_engine_v22_9.py
# 封装一个本地数据引擎：优先从 feather 加载，若不存在则回退到 CSV。

import os
from typing import Optional

import pandas as pd

from data_feather_loader_v22_9 import load_feather_kline
from local_data_engine import load_local_kline  # 复用你现有的 CSV 加载逻辑


class LocalDataEngineV22_9:
    def __init__(self, feather_dir: str = "data/feather", csv_base_dir: str = "data", exchange: str = "binance"):
        self.feather_dir = feather_dir
        self.csv_base_dir = csv_base_dir
        self.exchange = exchange

    def load_klines(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        # 1) 先尝试 feather
        try:
            return load_feather_kline(symbol, interval, days=days, base_dir=self.feather_dir)
        except FileNotFoundError:
            pass

        # 2) 回退到 CSV（沿用你现有的 V12/V22 逻辑）
        return load_local_kline(symbol, interval, days)
