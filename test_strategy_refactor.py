# test_strategy_refactor.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.strategies.strategy_factory import strategy_factory
import pandas as pd
import numpy as np

def test_strategy_factory():
    """测试策略工厂功能"""
    print("🧪 测试策略工厂...")
    
    # 1. 测试可用策略
    available = strategy_factory.get_available_strategies()
    print(f"✅ 可用策略: {available}")
    
    # 2. 测试创建SMA策略
    sma_config = {
        'name': '测试SMA策略',
        'symbols': ['BTC/USDT', 'ETH/USDT'],
        'parameters': {
            'sma_fast': 10,
            'sma_slow': 30
        }
    }
    
    try:
        sma_strategy = strategy_factory.create_strategy('SimpleMovingAverageStrategy', sma_config)
        print("✅ SMA策略创建成功")
        
        # 测试策略信息
        info = sma_strategy.get_strategy_info()
        print(f"📊 策略信息: {info}")
        
        # 测试信号计算
        test_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105],
            'high': [102, 103, 104, 105, 106, 107],
            'low': [98, 99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105, 106],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500]
        })
        
        signals = sma_strategy.calculate_signals(test_data)
        if not signals.empty:
            print(f"✅ 信号计算成功，最新信号: {signals['signal'].iloc[-1]}")
        else:
            print("⚠️  信号计算返回空数据")
            
    except Exception as e:
        print(f"❌ SMA策略测试失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 开始策略重构验证...")
    success = test_strategy_factory()
    
    if success:
        print("🎉 策略重构验证成功！")
    else:
        print("❌ 策略重构验证失败，需要调试")