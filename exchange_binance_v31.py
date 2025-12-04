# exchange_binance_v31.py
# 封装 Binance USDT 永续接口（ccxt）

import ccxt
import time
from typing import Dict, Any, Optional


class BinanceFuturesClient:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        default_type: str = "future",
    ):
        self.exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": default_type,  # "future" = USDT 永续
            },
        })

        # 测试网
        if testnet:
            self.exchange.set_sandbox_mode(True)

        self.markets = self.exchange.load_markets()

    # ---- 基本行情 ----
    def fetch_ohlcv(self, symbol: str, timeframe: str = "5m", limit: int = 200):
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        return self.exchange.fetch_ticker(symbol)

    # ---- 账户 & 持仓 ----
    def fetch_balance(self) -> Dict[str, Any]:
        # futures 账户余额
        return self.exchange.fetch_balance(params={"type": "future"})

    def fetch_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        返回指定symbol的持仓信息（简化版），如果没有持仓返回 None
        """
        positions = self.exchange.fetch_positions([symbol])
        for pos in positions:
            if pos.get("symbol") == symbol:
                # ccxt 标准字段：
                #   pos['contracts'], pos['side'], pos['entryPrice'], pos['unrealizedPnl']
                contracts = float(pos.get("contracts", 0) or 0)
                if contracts == 0:
                    return None
                return pos
        return None

    # ---- 下单 ----
    def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        side: "buy" / "sell"
        amount: 合约张数（币的数量，非USDT名义）
        """
        if params is None:
            params = {}
        order = self.exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=amount,
            params=params,
        )
        return order

    def create_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if params is None:
            params = {}
        order = self.exchange.create_order(
            symbol=symbol,
            type="limit",
            side=side,
            amount=amount,
            price=price,
            params=params,
        )
        return order

    def cancel_all_orders(self, symbol: str):
        try:
            self.exchange.cancel_all_orders(symbol)
        except Exception as e:
            print(f"[Binance] cancel_all_orders error for {symbol}: {e}")

    # ---- 工具 ----
    def get_last_price(self, symbol: str) -> float:
        ticker = self.fetch_ticker(symbol)
        return float(ticker["last"])

    def set_leverage(self, symbol: str, leverage: int):
        """
        设置某个币对的杠杆倍数
        """
        try:
            self.exchange.fapiPrivate_post_leverage({
                "symbol": symbol.replace("/", ""),
                "leverage": leverage,
            })
        except Exception as e:
            print(f"[Binance] 设置杠杆失败 {symbol}: {e}")
            # 某些 ccxt 版本可以用 unified params:
            # self.exchange.set_leverage(leverage, symbol, params={"marginMode": "isolated"})


if __name__ == "__main__":
    # 简单自测用，正式运行用 live_trade_engine 调用
    print("This module is intended to be used by live_trade_engine_v31_3.py")
