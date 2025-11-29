import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

BINANCE_URL = "https://api.binance.com/api/v3/klines"


# ================================================================
#                    真实市场数据接口（最终优化版）
# ================================================================
class RealMarketData:
    """真实市场数据接口（含 Binance API + 本地缓存 + 模拟数据）"""

    def __init__(self):
        self.cache = {}  # 避免重复下载
        self.session = requests.Session()

    # ------------------------------------------------------------
    # 🔹 方法1：下载 Binance 真实K线
    # ------------------------------------------------------------
    def get_recent_klines(self, symbol: str, interval="1h", days=30) -> pd.DataFrame:
        """
        下载 Binance K线数据，支持 days 天。
        """
        limit = min(days * 24, 1000)   # Binance 单次最多 1000 根

        # 将 BTC/USDT 转换为 Binance API 规范 BTCUSDT
        api_symbol = symbol.replace("/", "")

        url = f"{BINANCE_URL}?symbol={api_symbol}&interval={interval}&limit={limit}"

        try:
            r = self.session.get(url, timeout=5)
            data = r.json()

            if isinstance(data, dict) and "code" in data:
                print(f"⚠️ Binance返回错误: {data}")
                return pd.DataFrame()

            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close",
                "volume", "_1", "_2", "_3", "_4", "_5", "_6"
            ])

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["open"] = df["open"].astype(float)
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["close"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(float)

            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df.sort_values("timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)

            print(f"📥 下载真实K线成功: {symbol}, {len(df)} 行")
            return df

        except Exception as e:
            print(f"❌ 下载真实数据失败: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------
    # 🔹 方法2：生成模拟市场数据（备用）
    # ------------------------------------------------------------
    def _generate_fake_data(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        """生成趋势市场 + 随机波动的模拟数据"""

        base_prices = {
            "BTC/USDT": 50000,
            "ETH/USDT": 3000,
            "SOL/USDT": 150,
            "ADA/USDT": 0.5
        }

        base = base_prices.get(symbol, 100)

        prices = [base]
        for i in range(limit - 1):
            drift = np.random.normal(0, 0.002)
            noise = np.random.normal(0, 0.01)
            jump = np.random.normal(0, 0.04) if np.random.rand() < 0.03 else 0
            prices.append(prices[-1] * (1 + drift + noise + jump))

        df = pd.DataFrame({
            "timestamp": [datetime.now() - timedelta(minutes=5 * i) for i in range(limit)][::-1],
            "open": prices,
            "high": [p * (1 + np.random.rand() * 0.01) for p in prices],
            "low": [p * (1 - np.random.rand() * 0.01) for p in prices],
            "close": prices,
            "volume": np.random.randint(1000, 100000, size=limit)
        })

        print(f"📊 使用模拟市场数据: {symbol} ({limit}行)")
        return df

    # ------------------------------------------------------------
    # 🔹 方法3：smart_backtest 专用接口
    # ------------------------------------------------------------
    def load_for_smart_backtest(self, symbol: str, days: int) -> pd.DataFrame:
        """
        回测专用数据接口：尝试真实数据 → 否则 fallback 模拟数据
        """
        # 1. 尝试从真实市场拿数据
        df = self.get_recent_klines(symbol, interval="5m", days=days)

        if df is not None and not df.empty:
            return df

        # 2. 不行则 fallback 模拟数据
        print(f"⚠️ 使用 fallback 模拟数据: {symbol}")
        return self._generate_fake_data(symbol, limit=days * 24 * 12)


# ================================================================
#                     测试入口（可选）
# ================================================================
if __name__ == "__main__":
    rm = RealMarketData()
    df = rm.load_for_smart_backtest("BTC/USDT", 30)
    print(df.head())
    print(df.tail())
