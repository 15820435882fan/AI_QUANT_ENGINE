# local_data_engine_v22_9.py
import os
import pandas as pd
from data_feather_loader_v22_9 import load_feather_kline

class LocalDataEngineV22_9:
    def __init__(self, csv_dir="data/binance", feather_dir="data/feather"):
        self.csv_dir = csv_dir
        self.feather_dir = feather_dir

    def load(self, symbol: str, interval: str):
        """
        优先加载 feather，没有再加载 CSV
        """
        # 1) Feather exists?
        fname_feather = f"{symbol.replace('/', '')}-{interval}.feather"
        fpath_feather = os.path.join(self.feather_dir, fname_feather)
        if os.path.exists(fpath_feather):
            return load_feather_kline(symbol, interval, self.feather_dir)

        # 2) CSV fallback
        fname_csv = f"{interval}.csv"
        fpath_csv = os.path.join(self.csv_dir, symbol.replace('/', ''), fname_csv)

        if not os.path.exists(fpath_csv):
            raise FileNotFoundError(f"CSV 文件不存在: {fpath_csv}")

        df = pd.read_csv(fpath_csv)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        df = df.sort_index()

        return df
