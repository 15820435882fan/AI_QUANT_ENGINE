# fix_health_check.py
import sys
import importlib
from multi_strategy_manager_enhanced import MultiStrategyManagerEnhanced

def debug_health_check():
    """调试健康检查中的 'high' 错误"""
    print("🔧 调试健康检查问题...")
    
    # 测试1: 检查当前使用的模块版本
    manager = MultiStrategyManagerEnhanced()
    print(f"✅ MultiStrategyManagerEnhanced 版本: {id(manager)}")
    
    # 测试2: 检查方法是否存在
    if hasattr(manager, '_preprocess_market_data'):
        print("✅ _preprocess_market_data 方法存在")
    else:
        print("❌ _preprocess_market_data 方法缺失")
        return False
    
    # 测试3: 模拟健康检查的数据测试
    print("\n🧪 模拟健康检查数据测试...")
    test_data = {
        'close': [100, 101, 102, 103, 104],
        'volume': [1000, 2000, 1500, 3000, 2500]
    }
    test_df = pd.DataFrame(test_data)
    
    try:
        processed = manager._preprocess_market_data(test_df)
        print(f"✅ 数据预处理成功: {processed.shape}")
        print(f"✅ 数据列: {processed.columns.tolist()}")
        return True
    except Exception as e:
        print(f"❌ 数据预处理失败: {e}")
        return False

def force_reload_modules():
    """强制重新加载模块"""
    print("\n🔄 强制重新加载模块...")
    modules_to_reload = ['multi_strategy_manager_enhanced']
    
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
            print(f"✅ 重新加载: {module_name}")

if __name__ == "__main__":
    import pandas as pd
    
    force_reload_modules()
    success = debug_health_check()
    
    if success:
        print("\n🎉 调试完成！现在重新运行健康检查：")
        print("python system_health_check_final.py")
    else:
        print("\n🔧 需要进一步修复")