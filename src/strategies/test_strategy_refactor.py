# test_strategy_refactor.py - 更新测试
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
    print(f"✅ 新策略: {available['new_strategies']}")
    print(f"🔄 旧策略: {available['legacy_strategies']}")
    print(f"📊 全部策略: {available['all']}")
    
    # 2. 测试创建SMA策略（新式）
    if 'SimpleMovingAverageStrategy' in available['new_strategies']:
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
                'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
                'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
                'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
            })
            
            signals = sma_strategy.calculate_signals(test_data)
            if not signals.empty:
                print(f"✅ 信号计算成功，数据形状: {signals.shape}")
                print(f"📈 最新信号: {signals['signal'].iloc[-1]}")
            else:
                print("⚠️  信号计算返回空数据")
                
        except Exception as e:
            print(f"❌ SMA策略测试失败: {e}")
            return False
    
    # 3. 测试旧策略（如果有）
    if available['legacy_strategies']:
        legacy_strategy = available['legacy_strategies'][0]
        print(f"🧪 测试旧策略适配: {legacy_strategy}")
        
        try:
            legacy_config = {
                'name': f'测试{legacy_strategy}',
                'symbols': ['BTC/USDT']
            }
            
            legacy_instance = strategy_factory.create_strategy(legacy_strategy, legacy_config)
            print(f"✅ 旧策略适配成功: {legacy_strategy}")
            
        except Exception as e:
            print(f"⚠️  旧策略适配测试失败: {e}")
    
    return True

if __name__ == "__main__":
    print("🚀 开始策略重构验证...")
    success = test_strategy_factory()
    
    if success:
        print("🎉 策略重构验证成功！")
    else:
        print("❌ 策略重构验证失败，需要调试")