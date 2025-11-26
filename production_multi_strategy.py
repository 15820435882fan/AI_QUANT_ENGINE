# production_multi_strategy.py
#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
import numpy as np
from datetime import datetime
from src.strategies.strategy_orchestrator import TradingSignal, SignalType

class ProductionStrategy:
    """生产环境策略基类"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.name = self.config.get('name', '生产策略')
        self.symbols = self.config.get('symbols', ['BTC/USDT'])
        self.initialized = False
        
    async def initialize(self):
        """策略初始化"""
        self.initialized = True
        
    async def analyze(self, market_data):
        """分析市场数据"""
        raise NotImplementedError

class SmartRSIStrategy(ProductionStrategy):
    """智能RSI策略"""
    
    async def analyze(self, market_data):
        price = getattr(market_data, 'close', 0)
        
        # 简化版智能逻辑
        rsi_simulated = np.random.randint(20, 80)  # 模拟RSI值
        
        if rsi_simulated < 30:
            return TradingSignal(
                symbol=getattr(market_data, 'symbol', 'BTC/USDT'),
                signal_type=SignalType.BUY,
                price=price,
                strength=0.8,
                timestamp=datetime.now().timestamp(),
                reason=f"RSI超卖 ({rsi_simulated})"
            )
        elif rsi_simulated > 70:
            return TradingSignal(
                symbol=getattr(market_data, 'symbol', 'BTC/USDT'),
                signal_type=SignalType.SELL,
                price=price,
                strength=0.8,
                timestamp=datetime.now().timestamp(),
                reason=f"RSI超买 ({rsi_simulated})"
            )
        
        return None

class TrendFollowingStrategy(ProductionStrategy):
    """趋势跟踪策略"""
    
    async def analyze(self, market_data):
        price = getattr(market_data, 'close', 0)
        
        # 简化趋势判断
        trend_strength = np.random.uniform(-1, 1)
        
        if trend_strength > 0.3:
            return TradingSignal(
                symbol=getattr(market_data, 'symbol', 'BTC/USDT'),
                signal_type=SignalType.BUY,
                price=price,
                strength=abs(trend_strength),
                timestamp=datetime.now().timestamp(),
                reason=f"上升趋势 (强度: {trend_strength:.2f})"
            )
        elif trend_strength < -0.3:
            return TradingSignal(
                symbol=getattr(market_data, 'symbol', 'BTC/USDT'),
                signal_type=SignalType.SELL,
                price=price,
                strength=abs(trend_strength),
                timestamp=datetime.now().timestamp(),
                reason=f"下降趋势 (强度: {trend_strength:.2f})"
            )
        
        return None

class ProductionMultiStrategyManager:
    """生产环境多策略管理器"""
    
    def __init__(self):
        self.strategies = {
            'rsi': SmartRSIStrategy({'name': '智能RSI', 'symbols': ['BTC/USDT']}),
            'trend': TrendFollowingStrategy({'name': '趋势跟踪', 'symbols': ['BTC/USDT']})
        }
        self.signal_history = []
        
    async def initialize(self):
        """初始化所有策略"""
        for strategy in self.strategies.values():
            await strategy.initialize()
        print("✅ 所有策略初始化完成")
    
    async def analyze_market(self, market_data):
        """多策略市场分析"""
        signals = []
        
        for name, strategy in self.strategies.items():
            try:
                signal = await strategy.analyze(market_data)
                if signal:
                    signals.append((name, signal))
                    print(f"📊 {name}: {signal.signal_type.value} - {signal.reason}")
            except Exception as e:
                print(f"❌ {name} 策略错误: {e}")
        
        # 信号聚合
        if signals:
            final_signal = self.aggregate_signals(signals)
            self.signal_history.append(final_signal)
            return final_signal
        
        return None
    
    def aggregate_signals(self, signals):
        """聚合多个策略信号"""
        buy_strength = sum(s.strength for _, s in signals if s.signal_type == SignalType.BUY)
        sell_strength = sum(s.strength for _, s in signals if s.signal_type == SignalType.SELL)
        
        if buy_strength > sell_strength:
            return TradingSignal(
                symbol=signals[0][1].symbol,
                signal_type=SignalType.BUY,
                price=signals[0][1].price,
                strength=min(buy_strength, 1.0),
                timestamp=datetime.now().timestamp(),
                reason=f"多策略买入共识 (强度: {buy_strength:.2f})"
            )
        else:
            return TradingSignal(
                symbol=signals[0][1].symbol,
                signal_type=SignalType.SELL,
                price=signals[0][1].price,
                strength=min(sell_strength, 1.0),
                timestamp=datetime.now().timestamp(),
                reason=f"多策略卖出共识 (强度: {sell_strength:.2f})"
            )

async def production_test():
    """生产环境测试"""
    print("🏭 生产环境多策略系统测试")
    print("=" * 50)
    
    manager = ProductionMultiStrategyManager()
    await manager.initialize()
    
    # 生成测试数据
    prices = [50000]
    for i in range(100):
        change = np.random.normal(0, 0.015)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1000))
    
    signals = []
    
    for i, price in enumerate(prices):
        class MarketData:
            def __init__(self, price, idx):
                self.symbol = "BTC/USDT"
                self.close = price
                self.high = price * 1.01
                self.low = price * 0.99
                self.timestamp = datetime.now().timestamp() + idx * 3600
        
        market_data = MarketData(price, i)
        signal = await manager.analyze_market(market_data)
        
        if signal:
            signals.append(signal)
            print(f"🎯 交易信号: {signal.signal_type.value} @ {signal.price:.2f}")
            print(f"   强度: {signal.strength:.2f}, 原因: {signal.reason}")
            print("---")
    
    # 性能分析
    print(f"\n📈 性能报告:")
    print(f"   总数据点: {len(prices)}")
    print(f"   交易信号: {len(signals)}")
    print(f"   信号频率: {len(signals)/len(prices)*100:.1f}%")
    
    if signals:
        buy_signals = len([s for s in signals if s.signal_type == SignalType.BUY])
        sell_signals = len([s for s in signals if s.signal_type == SignalType.SELL])
        print(f"   买入信号: {buy_signals}")
        print(f"   卖出信号: {sell_signals}")
        
        # 简单回测
        if len(signals) >= 2:
            total_return = (signals[-1].price - signals[0].price) / signals[0].price * 100
            print(f"   模拟收益: {total_return:+.2f}%")

if __name__ == "__main__":
    asyncio.run(production_test())