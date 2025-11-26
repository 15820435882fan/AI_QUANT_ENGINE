# debug_constructor.py - 深度调试构造函数
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_constructor_issue():
    """深度调试构造函数问题"""
    print("🔍 深度调试构造函数...")
    
    try:
        # 1. 测试BaseStrategy直接创建
        from src.strategies.strategy_orchestrator import BaseStrategy
        
        print("✅ BaseStrategy导入成功")
        
        # 测试BaseStrategy
        base_config = {'name': '测试基础策略', 'parameters': {}}
        base_strategy = BaseStrategy(base_config)
        print("✅ BaseStrategy直接创建成功")
        
        # 2. 检查SMA策略的构造函数
        from src.strategies.simple_moving_average import SimpleMovingAverageStrategy
        print("✅ SMA策略导入成功")
        
        # 检查SMA的MRO（方法解析顺序）
        print(f"📊 SMA策略MRO: {SimpleMovingAverageStrategy.__mro__}")
        
        # 检查SMA的__init__方法签名
        import inspect
        sig = inspect.signature(SimpleMovingAverageStrategy.__init__)
        print(f"📊 SMA策略__init__签名: {sig}")
        
        # 检查BaseStrategy的__init__方法签名
        base_sig = inspect.signature(BaseStrategy.__init__)
        print(f"📊 BaseStrategy__init__签名: {base_sig}")
        
        return True
        
    except Exception as e:
        print(f"❌ 深度调试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sma_creation_detailed():
    """详细测试SMA创建过程"""
    print("\n🔍 详细测试SMA创建...")
    
    try:
        from src.strategies.simple_moving_average import SimpleMovingAverageStrategy
        
        # 详细配置
        config = {
            'name': '测试SMA策略',
            'symbols': ['BTC/USDT'],
            'parameters': {
                'sma_fast': 10,
                'sma_slow': 30
            }
        }
        
        print(f"📊 配置: {config}")
        print(f"📊 SMA类: {SimpleMovingAverageStrategy}")
        print(f"📊 SMA模块: {SimpleMovingAverageStrategy.__module__}")
        
        # 尝试创建实例
        sma_strategy = SimpleMovingAverageStrategy(config)
        print("✅ SMA策略创建成功！")
        
        info = sma_strategy.get_strategy_info()
        print(f"📊 策略信息: {info}")
        
        return True
        
    except Exception as e:
        print(f"❌ SMA创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 开始深度调试...")
    
    success1 = debug_constructor_issue()
    success2 = test_sma_creation_detailed()
    
    if success1 and success2:
        print("\n🎉 深度调试成功！")
    else:
        print("\n❌ 深度调试失败")