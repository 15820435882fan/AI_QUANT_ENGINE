from real_market_data_v3 import RealMarketData

m = RealMarketData()

df = m.get_recent_klines("BTC/USDT", "5m", 1)  # 仅 1 天，测试速度

print(df.head())
print("行数:", len(df))
