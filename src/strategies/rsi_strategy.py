# src/strategies/rsi_strategy.py
import pandas as pd
import numpy as np
from typing import Optional
from .strategy_orchestrator import BaseStrategy, TradingSignal, SignalType

class RSIStrategy(BaseStrategy):
    """真实的RSI策略"""
    
    def __init__(self, name: str, symbols: list, period: int = 14, oversold: int = 30, overbought: int = 70):
        config = {
            'name': name,
            'symbols': symbols,
            'parameters': {
                'period': period,
                'oversold': oversold,
                'overbought': overbought
            }
        }
        super().__init__(config)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.price_data = {symbol: [] for symbol in symbols}
    
    async def analyze(self, market_data) -> Optional[TradingSignal]:
        """RSI策略分析"""
        symbol = market_data.symbol
        
        # 获取收盘价
        close_price = self._extract_close_price(market_data)
        if close_price is None:
            return None
        
        # 更新价格数据
        if symbol not in self.price_data:
            self.price_data[symbol] = []
        
        self.price_data[symbol].append(close_price)
        
        # 保持数据长度
        if len(self.price_data[symbol]) > self.period * 2:
            self.price_data[symbol].pop(0)
        
        # 计算RSI
        if len(self.price_data[symbol]) >= self.period:
            rsi = self._calculate_rsi(self.price_data[symbol], self.period)
            
            if rsi < self.oversold:
                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=(self.oversold - rsi) / self.oversold,
                    price=close_price,
                    timestamp=market_data.timestamp,
                    reason=f"RSI超卖: {rsi:.1f}"
                )
            elif rsi > self.overbought:
                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL, 
                    strength=(rsi - self.overbought) / (100 - self.overbought),
                    price=close_price,
                    timestamp=market_data.timestamp,
                    reason=f"RSI超买: {rsi:.1f}"
                )
        
        return None
    
    def _calculate_rsi(self, prices: list, period: int) -> float:
        """计算RSI指标"""
        if len(prices) < period + 1:
            return 50.0  # 默认值
        
        # 计算价格变化
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # 计算平均增益和损失
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _extract_close_price(self, market_data):
        """提取收盘价"""
        data = market_data.data
        if isinstance(data, (list, tuple)) and len(data) >= 5:
            return float(data[4])
        elif isinstance(data, dict) and 'close' in data:
            return float(data['close'])
        return None