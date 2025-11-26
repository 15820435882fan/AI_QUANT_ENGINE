# optimized_multi_strategy_test.py
#!/usr/bin/env python3
import sys
import os
import asyncio
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_strategy_manager_enhanced import EnhancedMultiStrategyManager
from src.strategies.strategy_orchestrator import TradingSignal, SignalType
from data_compatibility_fix import DataCompatibility

def create_volatile_market_data(days=30, base_price=50000):
    """创建更波动的测试数据以触发更多信号"""
    print("📊 生成波动市场数据...")
    
    prices = []
    current = base_price
    timestamps = []
    
    # 创建更大的价格波动
    current_time = datetime.now() - timedelta(days=days)
    
    for i in range(days * 4):  # 4个数据点每天
        # 更大的波动率
        volatility = 0.02  # 2%波动
        trend = np.random.choice([-0.005, 0, 0.005])  # 随机趋势
        
        change = np.random.normal(trend, volatility)
        current = current * (1 + change)
        prices.append(current)
        
        timestamp = current_time + timedelta(hours=i*6)
        timestamps.append(timestamp.timestamp())
    
    return prices, timestamps

async def test_optimized_multi_strategy():
    """测试优化后的多策略"""
    print("🎯 测试优化版多策略组合")
    print("=" * 50)
    
    try:
        manager = EnhancedMultiStrategyManager(symbols=["BTC/USDT"])
        
        # 使用更波动的数据
        test_prices, test_timestamps = create_volatile_market_data(days=30, base_price=50000)
        
        print(f"📊 测试数据: {len(test_prices)} 个价格点")
        print(f"📈 价格范围: {min(test_prices):.2f} - {max(test_prices):.2f}")
        print(f"📊 波动率: {(max(test_prices)-min(test_prices))/test_prices[0]*100:.2f}%")
        
        signals = []
        
        # 测试不同市场状态
        regimes = [
            (0, "bull"),
            (len(test_prices)//3, "ranging"),
            (2*len(test_prices)//3, "trend")
        ]
        
        regime_index = 0
        
        for i, (price, timestamp) in enumerate(zip(test_prices, test_timestamps)):
            # 更新市场状态
            if regime_index < len(regimes) and i >= regimes[regime_index][0]:
                manager.update_market_regime(regimes[regime_index][1])
                print(f"🔄 切换到 {regimes[regime_index][1]} 市场")
                regime_index += 1
            
            # 使用兼容的数据格式
            market_data = DataCompatibility.create_compatible_data(price, timestamp)
            
            # 分析信号
            signal = await manager.analyze(market_data)
            
            if signal:
                signals.append(signal)
                print(f"🎯 信号 #{len(signals)}: {signal.signal_type.value} @ {signal.price:.2f}")
                print(f"   强度: {signal.strength:.3f}, 原因: {signal.reason}")
        
        # 性能分析
        print(f"\n📊 回测结果:")
        print(f"   总信号数: {len(signals)}")
        print(f"   信号频率: {len(signals)/len(test_prices)*100:.2f}%")
        
        if signals:
            buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
            sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
            print(f"   买入信号: {len(buy_signals)}")
            print(f"   卖出信号: {len(sell_signals)}")
            
            # 策略性能
            performance = manager.get_strategy_performance()
            print(f"\n🔧 各策略表现:")
            for strategy, stats in performance.items():
                print(f"   {strategy}: {stats['signal_count']} 个信号")
        
        return len(signals) > 0
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主测试函数"""
    print("🚀 开始优化版多策略测试")
    
    success = await test_optimized_multi_strategy()
    
    if success:
        print("\n🎉 优化测试成功！系统现在应该能生成交易信号了。")
    else:
        print("\n⚠️ 测试完成但信号较少，可能需要进一步调整策略参数。")

if __name__ == "__main__":
    asyncio.run(main())