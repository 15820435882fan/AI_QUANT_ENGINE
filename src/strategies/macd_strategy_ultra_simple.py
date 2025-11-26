# macd_strategy_ultra_simple.py
#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import asyncio
from collections import deque
from enum import Enum
from dataclasses import dataclass

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

@dataclass
class TradingSignal:
    symbol: str
    signal_type: SignalType
    strength: float
    price: float
    timestamp: float
    reason: str = ""
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class MACDStrategyUltraSimple:
    """MACD策略终极简化版 - 确保能生成信号"""
    
    def __init__(self, name: str, symbols: List[str], 
                 fast_period: int = 12, 
                 slow_period: int = 26, 
                 signal_period: int = 9):
        
        self.name = name
        self.symbols = symbols
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        # 数据存储
        self.price_data = {symbol: deque(maxlen=100) for symbol in symbols}
        self.signal_count = 0
        self.last_signal_time = {symbol: 0 for symbol in symbols}
    
    async def analyze(self, market_data) -> Optional[TradingSignal]:
        """终极简化版MACD分析"""
        symbol = market_data.symbol
        
        # 提取收盘价
        close_price = self._extract_close_price(market_data)
        if close_price is None:
            return None
        
        # 更新价格数据
        self.price_data[symbol].append(close_price)
        
        # 检查最小数据长度
        min_data_length = max(self.slow_period + self.signal_period, 10)
        if len(self.price_data[symbol]) < min_data_length:
            print(f"📈 数据积累中: {len(self.price_data[symbol])}/{min_data_length}")
            return None
        
        # 计算MACD指标
        macd_line, signal_line, histogram = self._calculate_macd(symbol)
        if not macd_line or len(macd_line) == 0:
            return None
        
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        current_histogram = histogram[-1]
        
        print(f"📊 {self.name}:")
        print(f"   价格: {close_price:.2f}")
        print(f"   MACD: {current_macd:.6f}")
        print(f"   Signal: {current_signal:.6f}")
        print(f"   Histogram: {current_histogram:.6f}")
        
        if len(histogram) >= 3:
            print(f"   Hist变化: {histogram[-3]:.6f} -> {histogram[-2]:.6f} -> {histogram[-1]:.6f}")
        
        # 超级简单的信号生成逻辑
        signal = self._generate_ultra_simple_signal(
            symbol, close_price, market_data.timestamp,
            current_macd, current_signal, current_histogram, histogram
        )
        
        return signal
    
    def _calculate_macd(self, symbol: str) -> Tuple[List[float], List[float], List[float]]:
        """计算MACD指标"""
        try:
            prices = list(self.price_data[symbol])
            
            if len(prices) < self.slow_period:
                return [], [], []
            
            # 使用pandas计算
            price_series = pd.Series(prices)
            
            # 计算EMA
            ema_fast = price_series.ewm(span=self.fast_period, adjust=False).mean()
            ema_slow = price_series.ewm(span=self.slow_period, adjust=False).mean()
            
            # MACD线和信号线
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
            histogram = macd_line - signal_line
            
            print(f"🔢 计算MACD: 数据范围 {min(prices):.2f} - {max(prices):.2f}")
            print(f"   EMA快线: {ema_fast.iloc[-1]:.2f}")
            print(f"   EMA慢线: {ema_slow.iloc[-1]:.2f}") 
            
            return macd_line.tolist(), signal_line.tolist(), histogram.tolist()
            
        except Exception as e:
            print(f"❌ MACD计算错误: {e}")
            return [], [], []
    
    def _generate_ultra_simple_signal(self, symbol: str, close_price: float, timestamp: float,
                                    current_macd: float, current_signal: float, 
                                    current_histogram: float, histogram: List[float]) -> Optional[TradingSignal]:
        """超级简单的信号生成逻辑"""
        
        if len(histogram) < 2:
            return None
        
        prev_histogram = histogram[-2]
        
        print(f"🔍 信号条件检查:")
        print(f"   MACD > Signal: {current_macd:.2f} > {current_signal:.2f} = {current_macd > current_signal}")
        print(f"   Histogram改善: {current_histogram:.2f} > {prev_histogram:.2f} = {current_histogram > prev_histogram}")
        print(f"   Histogram转正: {prev_histogram:.2f} <= 0 -> {current_histogram:.2f} > 0 = {prev_histogram <= 0 and current_histogram > 0}")
        
        # 条件1: 简单金叉 + 柱状图改善
        if current_macd > current_signal and current_histogram > prev_histogram:
            reason = f"简单金叉+改善 ({prev_histogram:.4f}→{current_histogram:.4f})"
            return self._create_signal(symbol, close_price, timestamp, SignalType.BUY, reason, 0.7)
        
        # 条件2: 柱状图负转正
        if prev_histogram <= 0 and current_histogram > 0:
            reason = f"柱状图负转正 ({prev_histogram:.4f}→{current_histogram:.4f})"
            return self._create_signal(symbol, close_price, timestamp, SignalType.BUY, reason, 0.8)
        
        # 条件3: 简单死叉 + 柱状图恶化
        if current_macd < current_signal and current_histogram < prev_histogram:
            reason = f"简单死叉+恶化 ({prev_histogram:.4f}→{current_histogram:.4f})"
            return self._create_signal(symbol, close_price, timestamp, SignalType.SELL, reason, 0.7)
        
        # 条件4: 柱状图正转负
        if prev_histogram >= 0 and current_histogram < 0:
            reason = f"柱状图正转负 ({prev_histogram:.4f}→{current_histogram:.4f})"
            return self._create_signal(symbol, close_price, timestamp, SignalType.SELL, reason, 0.8)
        
        print("💤 未满足任何信号条件")
        return None
    
    def _create_signal(self, symbol: str, price: float, timestamp: float, 
                      signal_type: SignalType, reason: str, strength: float) -> TradingSignal:
        """创建交易信号"""
        self.signal_count += 1
        
        print(f"🎯 {self.name} 信号 #{self.signal_count}: {signal_type.value}")
        print(f"   原因: {reason}")
        print(f"   强度: {strength:.2f}")
        print(f"   价格: {price:.2f}")
        
        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            price=price,
            timestamp=timestamp,
            reason=reason
        )
    
    def _extract_close_price(self, market_data):
        """提取收盘价"""
        try:
            if hasattr(market_data, 'data'):
                data = market_data.data
                if isinstance(data, (list, tuple)) and len(data) >= 5:
                    return float(data[4])
                elif isinstance(data, dict) and 'close' in data:
                    return float(data['close'])
            elif hasattr(market_data, 'close'):
                return float(market_data.close)
        except (ValueError, TypeError, IndexError) as e:
            print(f"❌ 收盘价提取错误: {e}")
        return None

# 测试函数
async def test_ultra_simple_macd():
    """测试终极简化版MACD策略"""
    print("🧪 测试终极简化版MACD策略...")
    print("=" * 60)
    
    strategy = MACDStrategyUltraSimple(
        name="MACD终极简化版",
        symbols=["BTC/USDT"],
        fast_period=12,
        slow_period=26,
        signal_period=9
    )
    
    # 创建测试数据
    def create_test_data():
        prices = []
        current = 50000
        
        # 下跌阶段
        for i in range(20):
            current = current * (1 + np.random.normal(-0.002, 0.001))
            prices.append(current)
        
        # 上涨阶段  
        for i in range(30):
            current = current * (1 + np.random.normal(0.002, 0.001))
            prices.append(current)
        
        return prices
    
    test_prices = create_test_data()
    
    print(f"📊 测试数据: {len(test_prices)} 个价格点")
    print(f"📈 价格范围: {min(test_prices):.2f} - {max(test_prices):.2f}")
    
    class SimpleMarketData:
        def __init__(self, price, timestamp):
            self.symbol = "BTC/USDT"
            self.data = [timestamp, price, price+50, price-50, price, 1000]
            self.timestamp = timestamp
    
    signals = []
    
    for i, price in enumerate(test_prices):
        market_data = SimpleMarketData(price, i)
        signal = await strategy.analyze(market_data)
        
        if signal:
            signals.append(signal)
            print(f"✅ 捕获信号 #{len(signals)}: {signal.signal_type.value} @ {signal.price:.2f}")
    
    print(f"\n🎉 测试完成 - 总信号: {len(signals)}")
    
    if not signals:
        print("❌ 仍然没有信号，需要更激进的策略!")

if __name__ == "__main__":
    asyncio.run(test_ultra_simple_macd())