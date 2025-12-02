# data_feather_loader_v22_9.py
import os
import pandas as pd

def load_feather_kline(symbol: str, interval: str, base_dir="data/feather"):
    """
    加载 Feather K线，自动解析 timestamp、排序、清洗
    """
    sym = symbol.replace("/", "").replace("_", "")
    fname = f"{sym}-{interval}.feather"
    fpath = os.path.join(base_dir, fname)

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Feather 文件不存在: {fpath}")

    df = pd.read_feather(fpath)

    # 统一字段处理
    if "timestamp" not in df.columns:
        raise ValueError(f"{fpath} 缺少 timestamp 字段")

    # 转换为 DatetimeIndex（毫秒 → ns）
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")

    # 保留主字段
    keep_cols = ["open", "high", "low", "close", "volume"]
    df = df[keep_cols]

    return df
