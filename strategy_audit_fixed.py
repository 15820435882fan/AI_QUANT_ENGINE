# strategy_audit_fixed.py
#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib.util
import inspect

def load_strategy_class(file_path, class_name):
    """动态加载策略类"""
    try:
        spec = importlib.util.spec_from_file_location("strategy_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        strategy_class = getattr(module, class_name)
        return strategy_class
    except Exception as e:
        print(f"   加载失败: {e}")
        return None

def audit_strategies_fixed():
    """修复版策略审计"""
    strategy_files = {
        'SimpleMovingAverageStrategy': 'src/strategies/simple_moving_average.py',
        'MACDStrategySmart': 'src/strategies/macd_strategy_smart.py',
        'BollingerBandsStrategy': 'src/strategies/bollinger_bands_strategy.py',
        'TurtleTradingStrategy': 'src/strategies/turtle_trading_strategy.py',
    }
    
    working_strategies = []
    broken_strategies = []
    
    for class_name, file_path in strategy_files.items():
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            continue
            
        print(f"🔍 检查策略: {class_name}")
        
        strategy_class = load_strategy_class(file_path, class_name)
        if not strategy_class:
            broken_strategies.append((class_name, file_path, "加载失败"))
            continue
        
        # 检查构造函数
        try:
            sig = inspect.signature(strategy_class.__init__)
            print(f"   构造函数: {sig}")
            
            # 尝试不同参数组合
            test_configs = [
                {'name': '测试策略', 'symbols': ['BTC/USDT']},
                {'config': {'name': '测试策略', 'symbols': ['BTC/USDT']}},
                {}
            ]
            
            success = False
            for config in test_configs:
                try:
                    if 'config' in sig.parameters:
                        instance = strategy_class(config=config)
                    else:
                        instance = strategy_class(**config)
                    success = True
                    print(f"   ✅ 使用参数 {config} 实例化成功")
                    working_strategies.append((class_name, file_path))
                    break
                except Exception as e:
                    print(f"   ❌ 参数 {config} 失败: {e}")
            
            if not success:
                broken_strategies.append((class_name, file_path, "所有参数组合都失败"))
                
        except Exception as e:
            print(f"   ❌ 检查失败: {e}")
            broken_strategies.append((class_name, file_path, str(e)))
    
    print(f"\n📊 审计结果:")
    print(f"   正常策略: {len(working_strategies)}")
    print(f"   异常策略: {len(broken_strategies)}")
    
    return working_strategies, broken_strategies

if __name__ == "__main__":
    working, broken = audit_strategies_fixed()