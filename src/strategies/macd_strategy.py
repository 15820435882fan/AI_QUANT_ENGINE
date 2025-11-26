# src/strategies/macd_strategy.py
#!/usr/bin/env python3
# 首先设置路径
import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from typing import Optional

try:
    from src.strategies.strategy_orchestrator import BaseStrategy, TradingSignal, SignalType
    from src.data.data_pipeline import MarketData, DataType
except ImportError as e:
    print(f"导入错误: {e}")
    # 临时定义以便测试
    from dataclasses import dataclass
    from enum import Enum
    
    class SignalType(Enum):
        BUY = "buy"
        SELL = "sell"
    
    @dataclass
    class TradingSignal:
        symbol: str
        signal_type: SignalType
        strength: float
        price: float
        timestamp: float
        reason: str = ""
    
    class BaseStrategy:
        def __init__(self, config):
            self.config = config

class MACDStrategy(BaseStrategy):
    """MACD趋势跟踪策略"""
    
    def __init__(self, name: str, symbols: list, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        config = {
            'name': name,
            'symbols': symbols,
            'parameters': {
                'fast_period': fast_period,
                'slow_period': slow_period,
                'signal_period': signal_period
            }
        }
        super().__init__(config)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.price_data = {symbol: [] for symbol in symbols}
        self.name = name
    
    async def analyze(self, market_data) -> Optional[TradingSignal]:
        """MACD策略分析"""
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
        max_length = self.slow_period + self.signal_period + 10
        if len(self.price_data[symbol]) > max_length:
            self.price_data[symbol] = self.price_data[symbol][-max_length:]
        
        # 计算MACD
        if len(self.price_data[symbol]) >= self.slow_period + self.signal_period:
            macd, signal, histogram = self._calculate_macd(self.price_data[symbol])
            
            if len(macd) == 0:
                return None
                
            current_macd = macd[-1]
            current_signal = signal[-1]
            current_histogram = histogram[-1]
            
            print(f"📊 {self.name}: MACD={current_macd:.4f}, Signal={current_signal:.4f}, Hist={current_histogram:.4f}")
            
            # 生成交易信号 - 更严格的逻辑
            if current_histogram > 0.001 and current_macd > current_signal and current_macd > 0:
                strength = min(abs(current_histogram) * 50, 0.8)
                print(f"🎯 {self.name} 买入! Hist: {current_histogram:.4f}")
                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=close_price,
                    timestamp=market_data.timestamp,
                    reason=f"MACD金叉, Hist: {current_histogram:.4f}"
                )
            elif current_histogram < -0.001 and current_macd < current_signal and current_macd < 0:
                strength = min(abs(current_histogram) * 50, 0.8)
                print(f"🎯 {self.name} 卖出! Hist: {current_histogram:.4f}")
                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=strength,
                    price=close_price,
                    timestamp=market_data.timestamp,
                    reason=f"MACD死叉, Hist: {current_histogram:.4f}"
                )
        
        return None
    
    def _calculate_macd(self, prices: list) -> tuple:
        """计算MACD指标"""
        if len(prices) < self.slow_period:
            return [], [], []
        
        try:
            # 转换为pandas Series以便计算EMA
            price_series = pd.Series(prices)
            
            # 计算EMA
            ema_fast = price_series.ewm(span=self.fast_period, adjust=False).mean()
            ema_slow = price_series.ewm(span=self.slow_period, adjust=False).mean()
            
            # 计算MACD线
            macd_line = ema_fast - ema_slow
            
            # 计算信号线
            signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
            
            # 计算柱状图
            histogram = macd_line - signal_line
            
            return macd_line.tolist(), signal_line.tolist(), histogram.tolist()
        except Exception as e:
            print(f"MACD计算错误: {e}")
            return [], [], []
    
    def _extract_close_price(self, market_data):
        """提取收盘价"""
        try:
            data = market_data.data
            if isinstance(data, (list, tuple)) and len(data) >= 5:
                return float(data[4])
            elif isinstance(data, dict) and 'close' in data:
                return float(data['close'])
            elif hasattr(market_data, 'close'):
                return float(market_data.close)
        except (ValueError, TypeError, IndexError) as e:
            print(f"收盘价提取错误: {e}")
        return None

# 测试函数
async def test_macd_strategy():
    """测试MACD策略"""
    print("🧪 测试MACD策略...")
    
    strategy = MACDStrategy(
        name="MACD测试",
        symbols=["BTC/USDT"],
        fast_period=12,
        slow_period=26,
        signal_period=9
    )
    
    # 创建测试数据 - 模拟明显的趋势
    test_prices = []
    current_price = 50000
    for i in range(100):
        # 模拟上涨趋势
        trend = 0.001  # 0.1% 的上涨趋势
        noise = np.random.normal(0, 0.0005)
        current_price = current_price * (1 + trend + noise)
        test_prices.append(current_price)
    
    from src.data.data_pipeline import MarketData, DataType
    import time
    
    signals_generated = 0
    for i, price in enumerate(test_prices[-30:]):  # 测试最后30个价格
        test_data = MarketData(
            symbol="BTC/USDT",
            data_type=DataType.OHLCV,
            data=[time.time(), price, price+50, price-50, price, 1000],
            timestamp=time.time()
        )
        
        signal = await strategy.analyze(test_data)
        if signal:
            signals_generated += 1
            print(f"✅ 信号 {signals_generated}: {signal.signal_type.value} - {signal.reason}")
    
    print(f"🎉 MACD策略测试完成, 生成 {signals_generated} 个信号")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_macd_strategy())