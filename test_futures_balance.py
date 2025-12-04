import ccxt
import yaml

cfg = yaml.safe_load(open("config_live_v31.yaml"))
ex = ccxt.binance({
    "apiKey": cfg["binance"]["apiKey"],
    "secret": cfg["binance"]["secret"],
    "enableRateLimit": True,
    "options": {
        "defaultType": "future"
    },
    "proxies": cfg["binance"].get("proxies", None)
})

print(ex.fetch_balance())
