# clean_local_data.py

import os
import pandas as pd

BASE = "data/binance"

def clean_symbol(sym):
    folder = os.path.join(BASE, sym)
    if not os.path.exists(folder):
        print("⛔ 路径不存在:", folder)
        return

    for file in os.listdir(folder):
        fpath = os.path.join(folder, file)
        if fpath.endswith(".csv"):
            print("🧹 清理:", fpath)

            df = pd.read_csv(fpath)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")

            before = len(df)
            df = df[~df.index.duplicated(keep="last")]
            after = len(df)

            df.to_csv(fpath)
            print(f"   去重完成：{before} → {after}")

if __name__ == "__main__":
    for sym in ["BTCUSDT", "ETHUSDT"]:
        clean_symbol(sym)
