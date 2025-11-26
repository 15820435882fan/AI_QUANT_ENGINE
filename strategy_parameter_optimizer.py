# strategy_parameter_optimizer.py
#!/usr/bin/env python3
import sys
import numpy as np
import os
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_strategy_manager_enhanced import EnhancedMultiStrategyManager
from data_compatibility_fix import DataCompatibility

class StrategyParameterOptimizer:
    """策略参数优化器"""
    
    def __init__(self):
        self.optimal_params = {
            "macd": {"fast_period": 6, "slow_period": 13, "signal_period": 5},
            "bollinger": {"period": 10, "std_dev": 1.5},
            "turtle": {"entry_period": 10, "exit_period": 5, "atr_period": 7},
            "sma": {"fast_period": 3, "slow_period": 8}
        }
    
    def create_optimized_manager(self, symbols):
        """创建优化参数的多策略管理器"""
        manager = EnhancedMultiStrategyManager(symbols)
        
        # 重新初始化策略使用优化参数
        manager.strategies["sma"].fast_period = self.optimal_params["sma"]["fast_period"]
        manager.strategies["sma"].slow_period = self.optimal_params["sma"]["slow_period"]
        
        manager.strategies["macd"].fast_period = self.optimal_params["macd"]["fast_period"]
        manager.strategies["macd"].slow_period = self.optimal_params["macd"]["slow_period"]
        manager.strategies["macd"].signal_period = self.optimal_params["macd"]["signal_period"]
        
        manager.strategies["bollinger"].period = self.optimal_params["bollinger"]["period"]
        manager.strategies["bollinger"].std_dev = self.optimal_params["bollinger"]["std_dev"]
        
        manager.strategies["turtle"].entry_period = self.optimal_params["turtle"]["entry_period"]
        manager.strategies["turtle"].exit_period = self.optimal_params["turtle"]["exit_period"]
        manager.strategies["turtle"].atr_period = self.optimal_params["turtle"]["atr_period"]
        
        print("🎯 策略参数已优化:")
        for strategy, params in self.optimal_params.items():
            print(f"   {strategy}: {params}")
            
        return manager

async def test_optimized_strategies():
    """测试优化后的策略"""
    print("🚀 测试优化参数策略")
    print("=" * 50)
    
    optimizer = StrategyParameterOptimizer()
    manager = optimizer.create_optimized_manager(["BTC/USDT"])
    
    # 创建更长期、更波动的测试数据
    def create_extended_data(days=180, base_price=50000):
        import numpy as np
        from datetime import datetime, timedelta
        
        prices = []
        current = base_price
        timestamps = []
        
        current_time = datetime.now() - timedelta(days=days)
        
        # 创建明显的趋势和波动
        for i in range(days * 6):  # 6个数据点每天
            # 更大的波动率
            if i % 50 == 0:  # 每50个点改变趋势
                trend = np.random.choice([-0.01, -0.005, 0, 0.005, 0.01])
            
            volatility = 0.03  # 3%波动
            change = np.random.normal(trend, volatility)
            current = current * (1 + change)
            prices.append(current)
            
            timestamp = current_time + timedelta(hours=i*4)
            timestamps.append(timestamp.timestamp())
        
        return prices, timestamps
    
    test_prices, test_timestamps = create_extended_data(days=180)
    print(f"📊 测试数据: {len(test_prices)} 个价格点 ({len(test_prices)//6} 天)")
    
    signals = []
    signal_count = 0
    
    for i, (price, timestamp) in enumerate(zip(test_prices, test_timestamps)):
        market_data = DataCompatibility.create_compatible_data(price, timestamp)
        
        # 每30个数据点切换市场状态
        if i % 30 == 0:
            regime = np.random.choice(["bull", "bear", "ranging", "trend"])
            manager.update_market_regime(regime)
        
        signal = await manager.analyze(market_data)
        
        if signal:
            signal_count += 1
            signals.append(signal)
            print(f"🎯 信号 #{signal_count}: {signal.signal_type.value} @ {signal.price:.2f}")
            print(f"   强度: {signal.strength:.3f}, 原因: {signal.reason}")
    
    print(f"\n📊 最终结果:")
    print(f"   总信号数: {len(signals)}")
    print(f"   信号频率: {len(signals)/len(test_prices)*100:.2f}%")
    
    # 策略性能统计
    performance = manager.get_strategy_performance()
    print(f"\n🔧 各策略表现:")
    total_signals = 0
    for strategy, stats in performance.items():
        print(f"   {strategy}: {stats['signal_count']} 个信号")
        total_signals += stats['signal_count']
    
    print(f"   策略总信号: {total_signals}")
    
    return len(signals) > 10  # 期望至少10个信号

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(test_optimized_strategies())
    if success:
        print("\n🎉 优化成功！策略现在能生成足够的交易信号。")
    else:
        print("\n⚠️ 信号仍然较少，可能需要进一步调整。")