# clean_local_data_v2.py
import os
import pandas as pd

BASE = "data/binance"

def clean_symbol(sym):
    folder = os.path.join(BASE, sym)
    print(f"\n=== 🧹 开始清理 {sym} ===")
    if not os.path.exists(folder):
        print("⛔ 路径不存在:", folder)
        return

    for file in os.listdir(folder):
        if not file.endswith(".csv"):
            continue

        fpath = os.path.join(folder, file)
        print(f"📁 处理文件: {fpath}")

        df = pd.read_csv(fpath)

        # 统一格式
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        else:
            # Binance CSV 的 timestamp 在第一列
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)

        # 设置 index
        df = df.set_index("timestamp")

        before = len(df)

        # 🔥 去重（保留最新）
        df = df[~df.index.duplicated(keep="last")]

        # 🔥 按时间排序（关键）
        df = df.sort_index()

        # 🔥 删除任何逆序或坏数据（timestamp 必须递增）
        df = df.loc[df.index.notnull()]
        df = df[df.index.to_series().diff().fillna(pd.Timedelta(milliseconds=1)) >= pd.Timedelta(0)]

        after = len(df)

        # 保存
        df.to_csv(fpath)
        print(f"   ✔ 去重+排序完成：{before} → {after}")

if __name__ == "__main__":
    for sym in ["BTCUSDT", "ETHUSDT"]:
        clean_symbol(sym)
