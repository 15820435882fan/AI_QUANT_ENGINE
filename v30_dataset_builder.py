# v30_dataset_builder.py
#
# Build supervised sequence dataset for V30 models from teacher strategies.
#
# Pipeline:
#   1) Load 1h OHLCV data via LocalDataEngineV22_9 or local_data_engine
#   2) Use teacher_trend_follow() to generate teacher actions and risk levels
#   3) Build sliding window sequences:
#         X: [num_samples, seq_len, feature_dim]
#         y_action: [num_samples]
#         y_risk: [num_samples]
#      where each sample corresponds to the last bar in the window.
#   4) Split into train / valid / test chronologically and save as .npz files.
#
# Usage example:
#   python v30_dataset_builder.py --symbol BTCUSDT --timeframe 1h --days 2920 --seq-len 128

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from v30_teacher_strategies import teacher_trend_follow, TeacherConfig

try:
    from local_data_engine_v22_9 import LocalDataEngineV22_9
except Exception:
    LocalDataEngineV22_9 = None

try:
    from local_data_engine import load_local_kline
except Exception:
    load_local_kline = None


def load_ohlc(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Load OHLC data using LocalDataEngineV22_9 or fallback to local_data_engine.load_local_kline."""
    if LocalDataEngineV22_9 is not None:
        engine = LocalDataEngineV22_9(feather_dir="data/feather")
        df = engine.load_klines(symbol, timeframe, days)
    elif load_local_kline is not None:
        df = load_local_kline(symbol, timeframe, days)
    else:
        raise RuntimeError("No data loader available. Please provide local_data_engine_v22_9 or local_data_engine.")
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="ignore")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.set_index("timestamp")
    df = df.sort_index()
    required_cols = ["open", "high", "low", "close"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    return df[required_cols].copy()


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Build feature columns for V30 supervised dataset.

    For now we reuse a subset of well-behaved indicators:
        - close normalized
        - ret_1, ret_6, ret_24
        - ma_fast_ratio, ma_slow_ratio
        - atr_ratio
        - vol_24
        - trend_slope
    """
    df = df.copy()
    close = df["close"]

    df["close_norm"] = close / close.iloc[0]
    df["ret_1"] = close.pct_change(1)
    df["ret_6"] = close.pct_change(6)
    df["ret_24"] = close.pct_change(24)

    # reuse teacher indicators if present
    if "ma_fast" in df.columns and "ma_slow" in df.columns:
        df["ma_fast_ratio"] = df["ma_fast"] / close - 1.0
        df["ma_slow_ratio"] = df["ma_slow"] / close - 1.0
    else:
        df["ma_fast_ratio"] = 0.0
        df["ma_slow_ratio"] = 0.0

    if "atr_ratio" not in df.columns:
        # fallback basic ATR ratio
        high = df["high"]
        low = df["low"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr_temp"] = tr.rolling(14).mean()
        df["atr_ratio"] = df["atr_temp"] / close
    if "vol_24" not in df.columns:
        df["ret_1_temp"] = close.pct_change()
        df["vol_24"] = df["ret_1_temp"].rolling(24).std()

    if "trend_slope" not in df.columns:
        df["close_shift_60"] = close.shift(60)
        df["trend_slope"] = (close - df["close_shift_60"]) / (df["close_shift_60"] + 1e-9)

    feat_cols: List[str] = [
        "close_norm",
        "ret_1",
        "ret_6",
        "ret_24",
        "ma_fast_ratio",
        "ma_slow_ratio",
        "atr_ratio",
        "vol_24",
        "trend_slope",
    ]

    df = df.dropna(subset=feat_cols).copy()

    return df, feat_cols


def make_sequences(
    df: pd.DataFrame,
    feat_cols: List[str],
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert teacher-annotated df into supervised sequences.

    Returns:
        X: [N, seq_len, F]
        y_action: [N]
        y_risk: [N]
        timestamps: [N] (numpy datetime64)
    """
    actions = df["action_teacher"].astype(int).values
    risks = df["risk_level"].astype(int).values
    feats = df[feat_cols].values
    ts = df.index.values

    n = len(df)
    if n < seq_len + 10:
        raise ValueError(f"Too few rows after feature/teacher filtering: {n}")

    X_list = []
    y_action_list = []
    y_risk_list = []
    ts_list = []

    for i in range(seq_len - 1, n):
        start = i + 1 - seq_len
        X_list.append(feats[start : i + 1])
        y_action_list.append(actions[i])
        y_risk_list.append(risks[i])
        ts_list.append(ts[i])

    X = np.stack(X_list, axis=0).astype(np.float32)
    y_action = np.array(y_action_list, dtype=np.int64)
    y_risk = np.array(y_risk_list, dtype=np.int64)
    timestamps = np.array(ts_list)

    return X, y_action, y_risk, timestamps


def split_and_save(
    X: np.ndarray,
    y_action: np.ndarray,
    y_risk: np.ndarray,
    timestamps: np.ndarray,
    symbol: str,
    timeframe: str,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    n = X.shape[0]
    train_end = int(n * 0.7)
    valid_end = int(n * 0.85)

    idx_train = slice(0, train_end)
    idx_valid = slice(train_end, valid_end)
    idx_test = slice(valid_end, n)

    base = f"v30_{symbol}_{timeframe}"

    np.savez_compressed(
        os.path.join(out_dir, base + "_train.npz"),
        X=X[idx_train],
        y_action=y_action[idx_train],
        y_risk=y_risk[idx_train],
        timestamps=timestamps[idx_train],
    )
    np.savez_compressed(
        os.path.join(out_dir, base + "_valid.npz"),
        X=X[idx_valid],
        y_action=y_action[idx_valid],
        y_risk=y_risk[idx_valid],
        timestamps=timestamps[idx_valid],
    )
    np.savez_compressed(
        os.path.join(out_dir, base + "_test.npz"),
        X=X[idx_test],
        y_action=y_action[idx_test],
        y_risk=y_risk[idx_test],
        timestamps=timestamps[idx_test],
    )


def build_dataset(
    symbol: str,
    timeframe: str = "1h",
    days: int = 2920,
    seq_len: int = 128,
    out_dir: str = "datasets",
) -> None:
    print(f"[V30] Loading data: symbol={symbol}, timeframe={timeframe}, days={days}")
    df_raw = load_ohlc(symbol, timeframe, days)

    print("[V30] Running teacher strategy...")
    teacher_cfg = TeacherConfig()
    df_teacher = teacher_trend_follow(df_raw, cfg=teacher_cfg)

    print("[V30] Building features...")
    df_feat, feat_cols = build_features(df_teacher)

    print("[V30] Building sequences...")
    X, y_action, y_risk, timestamps = make_sequences(df_feat, feat_cols, seq_len=seq_len)

    print(f"[V30] Dataset sizes: X={X.shape}, y_action={y_action.shape}, y_risk={y_risk.shape}")
    print("[V30] Splitting and saving...")
    split_and_save(
        X=X,
        y_action=y_action,
        y_risk=y_risk,
        timestamps=timestamps,
        symbol=symbol,
        timeframe=timeframe,
        out_dir=out_dir,
    )
    print(f"[V30] Done. Files saved to: {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="V30 dataset builder (teacher-based supervised sequences)")
    p.add_argument("--symbol", type=str, default="BTCUSDT", help="symbol, e.g., BTCUSDT")
    p.add_argument("--timeframe", type=str, default="1h", help="timeframe, e.g., 1h")
    p.add_argument("--days", type=int, default=2920, help="how many days of data to load")
    p.add_argument("--seq-len", type=int, default=128, help="sequence length (number of bars per sample)")
    p.add_argument("--out-dir", type=str, default="datasets", help="output directory for .npz files")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        symbol=args.symbol.upper(),
        timeframe=args.timeframe,
        days=args.days,
        seq_len=args.seq_len,
        out_dir=args.out_dir,
    )
