# local_data_engine.py
# -*- coding: utf-8 -*-

import os
import logging

from typing import List, Optional

import pandas as pd
def load_local_kline(symbol: str, interval: str, days: int):
    """
    完整修复版：
    - 兼容 timestamp 在列 或 index 的情况
    - 强制将 index 转成 DatetimeIndex
    - 保证切片最近 days 天不会报错
    """
    import os
    import pandas as pd

    base_dir = "data"
    exchange = "binance"
    sym_key = symbol.replace("/", "").upper()
    fpath = os.path.join(base_dir, exchange, sym_key, f"{interval}.csv")

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"❌ 本地数据不存在: {fpath}")

    df = pd.read_csv(fpath)

    # 统一处理 timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp")
    else:
        # index 模式
        df.index = pd.to_datetime(df.index, errors="coerce")

    # 必须丢掉无法解析的行
    df = df[~df.index.isna()]

    # ====== 关键修复：DatetimeIndex 才能切片 ======
    df.index = pd.DatetimeIndex(df.index)

    # 切片
    if days is not None and days > 0:
        end_ts = df.index.max()
        start_ts = end_ts - pd.Timedelta(days=days)
        df = df[df.index >= start_ts]

    return df


from real_market_data_v3 import RealMarketData

logger = logging.getLogger(__name__)


class LocalDataEngine:
    """
    V12 本地数据引擎：
    - 数据目录结构：
      base_dir / EXCHANGE / SYMBOL / INTERVAL.csv
      例如：data/binance/BTCUSDT/5m.csv
    - 支持：
      - 下载 & 覆盖缓存
      - 从缓存按天数切片
    """

    def __init__(self, base_dir: str = "data", exchange: str = "binance"):
        self.base_dir = base_dir
        self.exchange = exchange
        self.market = RealMarketData()

    def _symbol_key(self, symbol: str) -> str:
        # BTC/USDT -> BTCUSDT
        return symbol.replace("/", "").upper()

    def _file_path(self, symbol: str, interval: str) -> str:
        sym_key = self._symbol_key(symbol)
        return os.path.join(self.base_dir, self.exchange, sym_key, f"{interval}.csv")

    def ensure_dirs(self, symbol: str):
        path = self._file_path(symbol, "5m")
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

    # ------------------ V12-1：全量下载并覆盖缓存 ------------------
    def download_and_cache(
        self,
        symbol: str,
        interval: str,
        days: int,
        overwrite: bool = True,
    ) -> pd.DataFrame:
        """
        通过 RealMarketData 下载最近 days 天数据，并写入 CSV。
        """
        self.ensure_dirs(symbol)
        fpath = self._file_path(symbol, interval)

        logger.info(
            "📡 [LocalDataEngine] 下载并缓存: %s %s, days=%d -> %s",
            symbol, interval, days, fpath,
        )

        df = self.market.get_recent_klines(symbol, interval, days)

        # 保证 index 为 datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
            else:
                df.index = pd.to_datetime(df.index)

        df.to_csv(fpath)
        logger.info("✅ 已写入本地缓存: %s 行 -> %s", len(df), fpath)
        return df

    # ------------------ V12-2：从本地加载指定天数数据 ------------------
    def load_klines(
        self,
        symbol: str,
        interval: str,
        days: int,
        auto_download_if_missing: bool = True,
    ) -> pd.DataFrame:
        """
        从本地 CSV 加载最近 days 天的 K 线；
        如文件不存在且允许，则自动下载。
        """
        self.ensure_dirs(symbol)
        fpath = self._file_path(symbol, interval)

        if not os.path.exists(fpath):
            if not auto_download_if_missing:
                raise FileNotFoundError(f"本地数据不存在: {fpath}")
            logger.warning("⚠️ 本地文件缺失，将从交易所下载: %s", fpath)
            df = self.download_and_cache(symbol, interval, days)
            return df

        df = pd.read_csv(fpath)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        else:
            df.index = pd.to_datetime(df.index)

        # 根据时间切片最近 days 天
        if days is not None and days > 0:
            end_ts = df.index.max()
            start_ts = end_ts - pd.Timedelta(days=days)
            df = df[df.index >= start_ts]

        logger.info(
            "📥 [LocalDataEngine] 载入本地数据: %s %s, 天数=%d, 行数=%d",
            symbol, interval, days, len(df),
        )
        return df

    # ------------------ V12-3：批量下载辅助 ------------------
    def batch_download(
        self,
        symbols: List[str],
        intervals: List[str],
        days: int,
    ):
        for sym in symbols:
            for itv in intervals:
                try:
                    self.download_and_cache(sym, itv, days)
                except Exception as e:
                    logger.error("❌ 批量下载失败: %s %s: %s", sym, itv, e)
