# fixed_ai_optimizer_test.py
"""
修复AI优化器导入问题，基于现有代码结构
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

def find_ai_optimizer_class():
    """动态查找AI优化器类"""
    try:
        # 方法1: 尝试常见类名
        from ai_strategy_optimizer_enhanced import AIStrategyOptimizerEnhanced as Optimizer
        return Optimizer, "AIStrategyOptimizerEnhanced"
    except ImportError:
        pass
    
    try:
        # 方法2: 尝试其他可能类名
        from ai_strategy_optimizer_enhanced import AIStrategyOptimizer as Optimizer
        return Optimizer, "AIStrategyOptimizer"
    except ImportError:
        pass
    
    try:
        # 方法3: 查看模块属性
        import ai_strategy_optimizer_enhanced as module
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if hasattr(attr, '__name__') and 'Optimizer' in attr.__name__:
                return attr, attr.__name__
    except:
        pass
    
    return None, "未找到"

def main():
    print("🔧 修复AI优化器导入...")
    
    # 查找正确的类
    OptimizerClass, class_name = find_ai_optimizer_class()
    
    if OptimizerClass:
        print(f"✅ 找到优化器类: {class_name}")
        
        # 测试优化器
        from real_market_data import RealMarketData
        
        # 获取数据
        market_data = RealMarketData()
        data = market_data.get_historical_data('BTC/USDT', days=7, timeframe='1h')  # 先用7天快速测试
        
        print(f"📊 数据获取: {len(data) if data is not None else 0}条记录")
        
        if data is not None and len(data) > 0:
            # 初始化优化器
            optimizer = OptimizerClass()
            print("🚀 启动AI策略优化...")
            
            # 尝试优化
            try:
                result = optimizer.optimize_strategy(
                    strategy_class='SimpleMovingAverageStrategy',
                    data=data,
                    generations=3  # 快速测试
                )
                print(f"🎯 AI优化结果: {result}")
            except Exception as e:
                print(f"⚠️ 优化过程错误: {e}")
                print("尝试查看优化器方法...")
                print("优化器方法:", [method for method in dir(optimizer) if not method.startswith('_')])
        else:
            print("❌ 数据获取失败")
    else:
        print("❌ 未找到AI优化器类")
        print("📋 手动检查类名...")
        import ai_strategy_optimizer_enhanced as module
        print("模块内容:", [x for x in dir(module) if not x.startswith('_')])

if __name__ == "__main__":
    main()