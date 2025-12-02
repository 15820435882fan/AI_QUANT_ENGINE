import pandas as pd
from local_data_engine import LocalDataEngine
from structure_engine_v15 import analyze_structure

engine = LocalDataEngine(base_dir="data", exchange="binance")

df = engine.load_klines("BTC/USDT", "5m", days=30)
fractals, bis = analyze_structure(df)

print("BTC 5m 分型数量:", len(fractals))
print("BTC 5m 笔数量:", len(bis))

for b in bis[:5]:
    print(
        b.direction,
        b.start_time,
        "->",
        b.end_time,
        f"{b.start_price:.2f} -> {b.end_price:.2f}",
        f"bars={b.bars}, pct={b.length_pct*100:.2f}%",
    )
