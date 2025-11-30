# download_all_data.py
import logging
from local_data_engine import LocalDataEngine

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    engine = LocalDataEngine(base_dir="data", exchange="binance")
    symbols = ["BTC/USDT", "ETH/USDT"]
    intervals = ["1m", "5m", "15m", "1h", "4h"]
    days = 365

    engine.batch_download(symbols, intervals, days)
