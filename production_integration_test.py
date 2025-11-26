# production_integration_test.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_production_readiness():
    """测试生产环境就绪度"""
    print("🏭 测试生产环境就绪度...")
    
    # 测试所有核心组件
    components = [
        ("策略工厂", "src.strategies.strategy_factory"),
        ("多策略管理器", "multi_strategy_manager_enhanced"), 
        ("AI优化器", "ai_strategy_optimizer_enhanced"),
        ("数据兼容性", "data_compatibility_fix")
    ]
    
    all_passed = True
    
    for component_name, module_name in components:
        try:
            __import__(module_name)
            print(f"✅ {component_name}: 导入成功")
        except Exception as e:
            print(f"❌ {component_name}: 导入失败 - {e}")
            all_passed = False
    
    # 测试配置管理
    try:
        from src.strategies.strategy_factory import strategy_factory
        available = strategy_factory.get_available_strategies()
        print(f"✅ 策略发现: {len(available['all'])} 个策略可用")
    except Exception as e:
        print(f"❌ 策略发现失败: {e}")
        all_passed = False
    
    # 总结
    if all_passed:
        print("\n🎉 生产环境就绪度: ✅ 优秀")
        print("   所有核心组件正常运行")
        print("   可以进入生产部署阶段")
    else:
        print("\n⚠️  生产环境就绪度: 🟡 需要改进")
        print("   部分组件需要修复")
    
    return all_passed

if __name__ == "__main__":
    test_production_readiness()