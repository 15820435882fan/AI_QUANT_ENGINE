# real_market_data_v2.py
import time
import logging
from typing import Optional

import requests
import pandas as pd

logger = logging.getLogger(__name__)

BINANCE_BASE = "https://api.binance.com"


class RealMarketData:
    """
    V2 版 Binance 数据引擎：
    - 支持基于 days 自动分页抓取
    - 每次请求最多 1000 根K线，循环直到满足需求或无更多数据
    """

    def __init__(self, base_url: str = BINANCE_BASE, timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @staticmethod
    def _interval_to_ms(interval: str) -> int:
        """将 Binance interval 字符串转为毫秒."""
        mapping = {
            "1m": 1 * 60 * 1000,
            "3m": 3 * 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "30m": 30 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "2h": 2 * 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "6h": 6 * 60 * 60 * 1000,
            "8h": 8 * 60 * 60 * 1000,
            "12h": 12 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }
        if interval not in mapping:
            raise ValueError(f"暂不支持的 interval: {interval}")
        return mapping[interval]

    def _fetch_klines_batch(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        limit: int = 1000,
    ):
        """
        拉取一批 K 线（最多 1000 根）。
        start_time 为开盘时间起点（毫秒时间戳）。
        """
        params = {
            "symbol": symbol.replace("/", ""),
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if start_time is not None:
            params["startTime"] = int(start_time)

        url = f"{self.base_url}/api/v3/klines"
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data

    def get_recent_klines(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """
        获取最近 days 天的 K 线，自动多轮分页。
        - 使用 5m 时，days=30 对应理论 ~8640 根，days=60 对应 ~17280 根
        - 实际抓取时受限于 Binance 单次 1000 条，我们循环拉取并拼接
        """
        interval_ms = self._interval_to_ms(interval)
        bars_per_day = 24 * 60 * 60 * 1000 // interval_ms
        target_bars = days * bars_per_day

        now_ms = int(time.time() * 1000)
        start_time = now_ms - days * 24 * 60 * 60 * 1000

        logger.info(
            "📡 开始抓取 Binance K线: %s, interval=%s, target_days=%d, target_bars≈%d",
            symbol,
            interval,
            days,
            target_bars,
        )

        all_klines = []
        max_loops = 50  # 安全上限，避免死循环
        loops = 0
        current_start = start_time

        while len(all_klines) < target_bars and loops < max_loops:
            loops += 1
            try:
                batch = self._fetch_klines_batch(
                    symbol=symbol,
                    interval=interval,
                    start_time=current_start,
                    limit=1000,
                )
            except Exception as e:
                logger.error("❌ 拉取 %s K线失败: %s", symbol, e)
                break

            if not batch:
                logger.warning("⚠️ %s 没有更多 K线数据（batch 为空），提前结束", symbol)
                break

            all_klines.extend(batch)

            last_open_time = batch[-1][0]
            current_start = last_open_time + interval_ms

            logger.info(
                "📥 已拉取 %d 根K线 (%s), loops=%d",
                len(all_klines),
                symbol,
                loops,
            )

            # 如果时间已经逼近现在，也可以提前结束
            if current_start >= now_ms:
                break

        if not all_klines:
            logger.warning("⚠️ %s 未获取到任何K线数据", symbol)
            return pd.DataFrame()

        # 构建 DataFrame
        cols = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ]
        df = pd.DataFrame(all_klines, columns=cols)

        # 转换类型
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)

        float_cols = ["open", "high", "low", "close", "volume"]
        for c in float_cols:
            df[c] = df[c].astype(float)

        df = df[float_cols]  # 只保留主要价格字段

        # 去重 & 排序
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)

        # 如果多抓了，就取最后 target_bars 根（更靠近现在）
        if len(df) > target_bars:
            df = df.tail(target_bars)

        logger.info("✅ 最终 %s K线条数: %d (目标≈%d)", symbol, len(df), target_bars)

        return df
