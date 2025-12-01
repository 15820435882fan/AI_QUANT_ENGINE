# real_market_data_v3.py
# Binance 历史 K线下载 + 代理适配（用于国内环境）
# 你的 VPN（Mielink）基于 Clash 内核，混合代理端口为 26217

import requests
import logging
import time
import pandas as pd

logger = logging.getLogger(__name__)

class RealMarketData:
    def __init__(self):
        # 设置 Clash / Mielink 代理
        self.proxies = {
            "http": "http://127.0.0.1:26217",
            "https": "http://127.0.0.1:26217"
        }
        self.base = "https://api.binance.com"

    # -------------------- 单次请求 --------------------
    def fetch_klines_once(self, symbol, interval, startTime=None, limit=1000):
        url = f"{self.base}/api/v3/klines"
        params = {
            "symbol": symbol.replace("/", "").upper(),
            "interval": interval,
            "limit": limit,
        }
        if startTime:
            params["startTime"] = int(startTime)

        try:
            resp = requests.get(url, params=params, timeout=10, proxies=self.proxies)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("❌ 请求失败: %s %s: %s", symbol, interval, e)
            return None

    # -------------------- 分页抓取 --------------------
    def get_recent_klines(self, symbol, interval, days):
        target_bars = int(days * 24 * (60 // int(interval[:-1])))
        logger.info("📡 开始抓取 Binance K线: %s, %s, 目标=%d bars",
                    symbol, interval, target_bars)

        all_data = []
        end_time = int(time.time() * 1000)

        max_loops = 500
        loops = 0

        while len(all_data) < target_bars and loops < max_loops:
            loops += 1
            new_data = self.fetch_klines_once(
                symbol, interval, startTime=end_time - (limit := 1000)*60*60*1000
            )

            if not new_data:
                time.sleep(1)
                continue

            all_data.extend(new_data)
            end_time = new_data[0][0]

            logger.info("📥 已获取 %d 行 (%s %s)", len(all_data), symbol, interval)

            time.sleep(0.2)

        if not all_data:
            logger.warning("⚠️ 未抓取到数据: %s %s", symbol, interval)
            return pd.DataFrame()

        df = pd.DataFrame(
            all_data,
            columns=[
                "timestamp","open","high","low","close",
                "volume","close_time","quote","trades",
                "taker_base","taker_quote","ignore"
            ]
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)

        return df[["open","high","low","close","volume"]]
