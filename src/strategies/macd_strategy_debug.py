# macd_strategy_debug.py
#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np
from typing import Optional
import asyncio

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

try:
    from src.strategies.strategy_orchestrator import BaseStrategy, TradingSignal, SignalType
    from src.data.data_pipeline import MarketData, DataType
except ImportError:
    # 临时定义
    from enum import Enum
    from dataclasses import dataclass
    
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

class MACDStrategyDebug(BaseStrategy):
    """MACD策略调试版 - 更宽松的信号条件"""
    
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
        self.signal_count = 0
    
    async def analyze(self, market_data) -> Optional[TradingSignal]:
        """MACD策略分析 - 调试版本"""
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
        
        print(f"📈 数据长度: {len(self.price_data[symbol])}, 需要: {self.slow_period + self.signal_period}")
        
        # 计算MACD
        if len(self.price_data[symbol]) >= self.slow_period + self.signal_period:
            macd, signal, histogram = self._calculate_macd(self.price_data[symbol])
            
            if len(macd) == 0:
                print("❌ MACD计算返回空列表")
                return None
                
            current_macd = macd[-1]
            current_signal = signal[-1]
            current_histogram = histogram[-1]
            
            print(f"📊 {self.name}:")
            print(f"   价格: {close_price:.2f}")
            print(f"   MACD: {current_macd:.6f}")
            print(f"   Signal: {current_signal:.6f}") 
            print(f"   Histogram: {current_histogram:.6f}")
            print(f"   数据点: {len(self.price_data[symbol])}")
            
            # 调试信息 - 显示最近几个值的变化
            if len(histogram) >= 3:
                print(f"   Hist变化: {histogram[-3]:.6f} -> {histogram[-2]:.6f} -> {histogram[-1]:.6f}")
            
            # 更宽松的信号条件
            signal_generated = False
            reason = ""
            
            # 买入条件：柱状图转正且MACD上穿信号线
            if (current_histogram > -0.0001 and  # 几乎为正或为正
                current_macd > current_signal and 
                len(histogram) >= 2 and 
                histogram[-2] <= 0 and histogram[-1] > 0):  # 柱状图由负转正
                
                strength = min(abs(current_histogram) * 100, 0.9)
                reason = f"MACD金叉, Hist由负转正: {histogram[-2]:.6f} -> {histogram[-1]:.6f}"
                signal_generated = True
                action = SignalType.BUY
                
            # 卖出条件：柱状图转负且MACD下穿信号线  
            elif (current_histogram < 0.0001 and  # 几乎为负或为负
                  current_macd < current_signal and
                  len(histogram) >= 2 and 
                  histogram[-2] >= 0 and histogram[-1] < 0):  # 柱状图由正转负
                  
                strength = min(abs(current_histogram) * 100, 0.9)
                reason = f"MACD死叉, Hist由正转负: {histogram[-2]:.6f} -> {histogram[-1]:.6f}"
                signal_generated = True
                action = SignalType.SELL
            
            if signal_generated:
                self.signal_count += 1
                print(f"🎯 {self.name} 信号 #{self.signal_count}: {action.value}")
                print(f"   原因: {reason}")
                print(f"   强度: {strength:.2f}")
                
                return TradingSignal(
                    symbol=symbol,
                    signal_type=action,
                    strength=strength,
                    price=close_price,
                    timestamp=market_data.timestamp,
                    reason=reason
                )
            else:
                print("💤 未满足信号条件")
        
        return None
    
    def _calculate_macd(self, prices: list) -> tuple:
        """计算MACD指标 - 增强调试"""
        if len(prices) < self.slow_period:
            print(f"❌ 数据不足: {len(prices)} < {self.slow_period}")
            return [], [], []
        
        try:
            # 转换为pandas Series
            price_series = pd.Series(prices)
            
            print(f"🔢 计算MACD: 数据范围 {price_series.min():.2f} - {price_series.max():.2f}")
            
            # 计算EMA - 使用更精确的方法
            ema_fast = price_series.ewm(span=self.fast_period, adjust=False).mean()
            ema_slow = price_series.ewm(span=self.slow_period, adjust=False).mean()
            
            # 计算MACD线
            macd_line = ema_fast - ema_slow
            
            # 计算信号线
            signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
            
            # 计算柱状图
            histogram = macd_line - signal_line
            
            # 调试输出
            print(f"📐 MACD计算完成:")
            print(f"   EMA快线: {ema_fast.iloc[-1]:.2f}")
            print(f"   EMA慢线: {ema_slow.iloc[-1]:.2f}") 
            print(f"   MACD线: {macd_line.iloc[-1]:.6f}")
            print(f"   信号线: {signal_line.iloc[-1]:.6f}")
            print(f"   柱状图: {histogram.iloc[-1]:.6f}")
            
            return macd_line.tolist(), signal_line.tolist(), histogram.tolist()
            
        except Exception as e:
            print(f"❌ MACD计算错误: {e}")
            return [], [], []
    
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

# 创建有明显趋势的测试数据
def create_trending_data():
    """创建有明显趋势的测试数据"""
    prices = []
    current = 50000
    
    # 先下跌趋势
    for i in range(20):
        current = current * (1 + np.random.normal(-0.002, 0.001))
        prices.append(current)
    
    # 然后上涨趋势
    for i in range(30):
        current = current * (1 + np.random.normal(0.0015, 0.001))
        prices.append(current)
    
    return prices

async def test_macd_debug():
    """测试MACD调试版本"""
    print("🧪 测试MACD调试版本...")
    print("=" * 60)
    
    strategy = MACDStrategyDebug(
        name="MACD调试",
        symbols=["BTC/USDT"],
        fast_period=12,
        slow_period=26, 
        signal_period=9
    )
    
    # 使用有明显趋势的测试数据
    test_prices = create_trending_data()
    
    print(f"📊 测试数据: {len(test_prices)} 个价格点")
    print(f"📈 价格范围: {min(test_prices):.2f} - {max(test_prices):.2f}")
    
    # 创建市场数据对象
    class SimpleMarketData:
        def __init__(self, price, timestamp):
            self.symbol = "BTC/USDT"
            self.data = [timestamp, price, price+50, price-50, price, 1000]
            self.timestamp = timestamp
    
    signals = []
    
    # 逐步喂数据，模拟实时交易
    for i, price in enumerate(test_prices):
        market_data = SimpleMarketData(price, i)
        signal = await strategy.analyze(market_data)
        
        if signal:
            signals.append(signal)
            print(f"✅ 捕获信号 #{len(signals)}: {signal.signal_type.value} @ {signal.price:.2f}")
            print(f"   原因: {signal.reason}")
    
    print(f"\n🎉 MACD调试测试完成")
    print(f"📨 总生成信号: {len(signals)}")
    print(f"📊 测试数据趋势: 开始 {test_prices[0]:.2f} -> 结束 {test_prices[-1]:.2f}")
    
    if signals:
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        print(f"🛒 买入信号: {len(buy_signals)}")
        print(f"🏪 卖出信号: {len(sell_signals)}")
    else:
        print("❌ 未生成任何信号，需要进一步调试")

if __name__ == "__main__":
    asyncio.run(test_macd_debug())