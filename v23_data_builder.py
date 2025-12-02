# v23_data_builder.py
# 一次性重下 & 重采样 & 对齐 BTC/ETH/BNB/SOL 的 1m/5m/15m/1h/4h 数据
# 使用已有的 LocalDataEngine 下载，再用 pandas 从 1m 统一重采样

import os
import logging
from typing import List

import pandas as pd
from local_data_engine import LocalDataEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("V23DataBuilder")

# === 配置区 ===
BASE_DIR = "data"
EXCHANGE = "binance"

SYMBOLS: List[str] = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
]

BASE_INTERVAL = "1m"
TARGET_INTERVALS = ["5m", "15m", "1h", "4h"]

# 尽量覆盖 2017~至今，给一个足够大的天数
DAYS = 365 * 8  # 8 年


def symbol_to_dir(symbol: str) -> str:
    """
    将 'BTC/USDT' -> 'BTCUSDT'
    与 LocalDataEngine 当前的目录命名保持一致
    """
    return symbol.replace("/", "")


def csv_path(symbol: str, interval: str) -> str:
    """
    构造 CSV 路径：data/binance/BTCUSDT/5m.csv
    """
    sym_dir = symbol_to_dir(symbol)
    return os.path.join(BASE_DIR, EXCHANGE, sym_dir, f"{interval}.csv")


def ensure_parent_dir(fpath: str) -> None:
    d = os.path.dirname(fpath)
    os.makedirs(d, exist_ok=True)


def download_1m_data(engine: LocalDataEngine) -> None:
    """
    第一步：用 LocalDataEngine 从交易所下载 1m 原始 K 线
    """
    logger.info("=== Step 1: 下载 1m 原始 K 线（覆盖 8 年） ===")
    for sym in SYMBOLS:
        try:
            logger.info("下载 %s %s, days=%s", sym, BASE_INTERVAL, DAYS)
            engine.download_and_cache(sym, BASE_INTERVAL, DAYS)
        except Exception as e:
            logger.error("❌ 下载失败: %s %s: %s", sym, BASE_INTERVAL, e)


def resample_from_1m(symbol: str) -> None:
    """
    第二步：从 1m 统一重采样出 5m / 15m / 1h / 4h
    """
    base_f = csv_path(symbol, BASE_INTERVAL)
    if not os.path.exists(base_f):
        logger.error("⚠️ 1m 基础文件不存在: %s, 跳过该币种", base_f)
        return

    logger.info("=== 从 1m 重采样多周期: %s ===", symbol)
    df = pd.read_csv(base_f)

    if "timestamp" not in df.columns:
        raise ValueError(f"{base_f} 缺少 timestamp 列")

    # 转成 DatetimeIndex
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")

    # 去除明显异常行（如 open/high/low/close 全为 NaN）
    df = df[["open", "high", "low", "close", "volume"]]
    df = df.dropna(how="all")

    if len(df) == 0:
        logger.error("⚠️ 1m 数据为空: %s", base_f)
        return

    logger.info(
        "%s 1m 基础数据：rows=%d, start=%s, end=%s",
        symbol,
        len(df),
        df.index.min(),
        df.index.max(),
    )

    # 定义 resample 频率映射
    freq_map = {
        "5m": "5T",
        "15m": "15T",
        "1h": "1H",
        "4h": "4H",
    }

    for itv in TARGET_INTERVALS:
        if itv not in freq_map:
            logger.warning("未支持的目标周期: %s, 跳过", itv)
            continue

        rule = freq_map[itv]
        logger.info("重采样 %s -> %s (%s)", BASE_INTERVAL, itv, rule)

        df_res = df.resample(rule).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )

        # 没有任何成交的时间段：open/close 等可能为 NaN
        # 这里的策略：只要 close 为 NaN，就整体丢弃该 K 线
        df_res = df_res.dropna(subset=["close"])

        out_f = csv_path(symbol, itv)
        ensure_parent_dir(out_f)
        df_res.to_csv(out_f, index_label="timestamp")

        logger.info(
            "写入 %s: rows=%d, start=%s, end=%s",
            out_f,
            len(df_res),
            df_res.index.min(),
            df_res.index.max(),
        )


def quick_check(symbol: str) -> None:
    """
    第三步：快速检查 5m 数据的完整性，打印一些统计信息
    """
    f_5m = csv_path(symbol, "5m")
    if not os.path.exists(f_5m):
        logger.warning("⚠️ 5m 文件不存在，跳过 quick_check: %s", f_5m)
        return

    df = pd.read_csv(f_5m)
    if "timestamp" not in df.columns:
        logger.warning("⚠️ 5m 文件缺少 timestamp 列: %s", f_5m)
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df["delta"] = df["timestamp"].diff()

    total = len(df)
    gaps = df["delta"].value_counts().sort_index(ascending=False).head(5)

    logger.info(
        "[CHECK] %s 5m: rows=%d, start=%s, end=%s",
        symbol,
        total,
        df["timestamp"].min(),
        df["timestamp"].max(),
    )
    logger.info("[CHECK] %s 5m: delta 分布 Top5:\n%s", symbol, gaps)


def main():
    engine = LocalDataEngine(base_dir=BASE_DIR, exchange=EXCHANGE)

    # Step 1: 重下 1m 数据
    download_1m_data(engine)

    # Step 2: 从 1m 重采样多周期
    for sym in SYMBOLS:
        resample_from_1m(sym)

    # Step 3: 快速检查 5m 完整性
    for sym in SYMBOLS:
        quick_check(sym)


if __name__ == "__main__":
    main()
