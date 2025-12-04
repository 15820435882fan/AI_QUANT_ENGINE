import ccxt
from pprint import pprint

ex = ccxt.binance({
    "apiKey": "cofCiTIiAr1gnBtoMW4dZ3a3EcyJ2ii9bM9OqTQfS7f5lZVMpMfKW8IWiT0Y1a4E",
    "secret": "TVdUrjwvXr33sBIVoQkIq3p03htNLHFtnqlkFXIr3WBGIzEYp8qbguZDe3eSE017",
    "enableRateLimit": True,
    "options": {
        "defaultType": "future"  # 添加这个选项
    }
})

# 尝试不同的balance获取方式
print("=== 方法1: fetch_balance ===")
try:
    balance = ex.fetch_balance(params={"type": "future"})
    pprint(balance)
except Exception as e:
    print("错误：", e)

print("\n=== 方法2: fetch_balance (不加参数) ===")
try:
    balance2 = ex.fetch_balance()
    pprint(balance2)
except Exception as e:
    print("错误：", e)

print("\n=== 方法3: fetch_positions ===")
try:
    pos = ex.fetch_positions()
    pprint(pos)
except Exception as e:
    print("错误：", e)

print("\n=== 方法4: fetch_position (单个) ===")
try:
    # 先获取交易对
    markets = ex.load_markets()
    symbol = list(markets.keys())[0]  # 取第一个交易对
    position = ex.fetch_position(symbol)
    pprint(position)
except Exception as e:
    print("错误：", e)

print("\n=== 检查支持的端点 ===")
print("ex.has['fetchBalance']:", ex.has['fetchBalance'])
print("ex.has['fetchPositions']:", ex.has['fetchPositions'])
print("ex.has['fetchPosition']:", ex.has['fetchPosition'])
