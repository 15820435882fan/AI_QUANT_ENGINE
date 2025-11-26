# test_ai_optimizer_with_new_architecture.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.strategies.strategy_factory import strategy_factory

def test_ai_optimizer_integration():
    """测试AI优化器与新架构的集成"""
    print("🧪 测试AI优化器与新策略架构集成...")
    
    # 获取所有可用策略
    available = strategy_factory.get_available_strategies()
    print(f"📊 可用策略: {available['all']}")
    
    # 测试策略参数验证
    test_cases = [
        {
            'strategy': 'SimpleMovingAverageStrategy',
            'valid_params': {'sma_fast': 10, 'sma_slow': 30},
            'invalid_params': {'sma_fast': 10}  # 缺少sma_slow
        },
        {
            'strategy': 'MACDStrategySmart', 
            'valid_params': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            'invalid_params': {'fast_period': 12}
        }
    ]
    
    for test_case in test_cases:
        strategy_type = test_case['strategy']
        
        # 测试有效参数
        valid_config = {
            'name': f'测试{strategy_type}',
            'parameters': test_case['valid_params']
        }
        
        try:
            strategy = strategy_factory.create_strategy(strategy_type, valid_config)
            is_valid = strategy.validate_parameters()
            print(f"✅ {strategy_type} 有效参数测试: {is_valid}")
        except Exception as e:
            print(f"❌ {strategy_type} 有效参数测试失败: {e}")
        
        # 测试无效参数
        invalid_config = {
            'name': f'测试{strategy_type}',
            'parameters': test_case['invalid_params']
        }
        
        try:
            strategy = strategy_factory.create_strategy(strategy_type, invalid_config)
            is_valid = strategy.validate_parameters()
            print(f"⚠️ {strategy_type} 无效参数测试: {is_valid} (期望: False)")
        except Exception as e:
            print(f"✅ {strategy_type} 无效参数正确拒绝: {e}")

if __name__ == "__main__":
    test_ai_optimizer_integration()