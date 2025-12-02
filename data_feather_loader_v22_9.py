# data_feather_loader_v22_9.py
# Feather 优先的数据加载器：自动识别时间列（date / datetime / time / timestamp 等），
# 并统一输出包含 [open, high, low, close, volume] 的 DataFrame（DatetimeIndex）。

import os
from typing import Optional, List

import pandas as pd


def _symbol_to_feather_name(symbol: str, interval: str) -> str:
    """将 BTCUSDT / BTC/USDT 映射为 BTC_USDT-15m.feather 这种命名。"""
    sym = symbol.replace("/", "").upper()
    if sym.endswith("USDT"):
        base = sym[:-4]
        pair = f"{base}_USDT"
    else:
        pair = sym
    return f"{pair}-{interval}.feather"


def _detect_time_column(df: pd.DataFrame, fpath: str) -> str:
    """在 df 中自动寻找时间列名。优先级：
    timestamp > date > datetime > time > open_time > t
    """
    candidates: List[str] = [
        "timestamp", "date", "datetime", "time", "open_time", "t",
    ]
    lower_cols = {c.lower(): c for c in df.columns}
    for key in candidates:
        if key in lower_cols:
            return lower_cols[key]
    raise ValueError(f"{fpath} 中未找到可识别的时间列（timestamp/date/datetime/time/open_time/t）")


def _standardize_ohlcv_columns(df: pd.DataFrame, fpath: str) -> pd.DataFrame:
    """将各种可能的列名映射为标准的 open/high/low/close/volume。"""
    col_map = {}
    lower_cols = {c.lower(): c for c in df.columns}

    def pick(names: List[str], target: str):
        for name in names:
            if name in lower_cols:
                col_map[lower_cols[name]] = target
                return

    pick(["open", "o"], "open")
    pick(["high", "h"], "high")
    pick(["low", "l"], "low")
    pick(["close", "c"], "close")
    pick(["volume", "vol", "v"], "volume")

    df = df.rename(columns=col_map)

    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{fpath} 缺少必要列: {col}")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    return df[["open", "high", "low", "close", "volume"]].copy()


def load_feather_kline(
    symbol: str,
    interval: str,
    days: Optional[int] = None,
    base_dir: str = "data/feather",
) -> pd.DataFrame:
    """从 feather 文件加载 K 线。

    - 自动识别时间列（date/datetime/time/timestamp/...）
    - 将其转换为 DatetimeIndex
    - 将 OHLCV 列映射为标准 [open, high, low, close, volume]
    - 可选按最近 days 天进行时间切片
    """
    fname = _symbol_to_feather_name(symbol, interval)
    fpath = os.path.join(base_dir, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Feather 数据不存在: {fpath}")

    df = pd.read_feather(fpath)
    if df.empty:
        raise ValueError(f"Feather 数据为空: {fpath}")

    # 自动识别时间列
    time_col = _detect_time_column(df, fpath)

    # 转为 DatetimeIndex
    ts = df[time_col]
    if pd.api.types.is_numeric_dtype(ts):
        # 通常是毫秒时间戳
        df[time_col] = pd.to_datetime(ts, unit="ms", errors="coerce")
    else:
        df[time_col] = pd.to_datetime(ts, errors="coerce")

    df = df.dropna(subset=[time_col]).sort_values(time_col)
    df = df.set_index(time_col)

    # 统一 OHLCV 列
    df_ohlcv = _standardize_ohlcv_columns(df, fpath)

    # 按最近 days 切片
    if days is not None and days > 0 and len(df_ohlcv) > 0:
        end_time = df_ohlcv.index.max()
        start_time = end_time - pd.Timedelta(days=days)
        df_ohlcv = df_ohlcv[df_ohlcv.index >= start_time]

    return df_ohlcv
