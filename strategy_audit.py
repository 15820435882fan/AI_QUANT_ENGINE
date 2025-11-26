# strategy_audit.py
#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import inspect
from src.strategies.strategy_orchestrator import BaseStrategy

def audit_strategies():
    """审计所有策略文件，检查构造函数和性能"""
    strategy_files = [
        'src/strategies/simple_moving_average.py',
        'src/strategies/macd_strategy_smart.py', 
        'src/strategies/bollinger_bands_strategy.py',
        'src/strategies/turtle_trading_strategy.py',
        'src/strategies/market_regime_detector.py'
    ]
    
    working_strategies = []
    broken_strategies = []
    
    for file_path in strategy_files:
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            continue
            
        try:
            # 动态导入策略类
            module_name = file_path.replace('/', '.').replace('.py', '')
            module = __import__(module_name, fromlist=['*'])
            
            # 查找策略类
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseStrategy) and 
                    obj != BaseStrategy):
                    
                    print(f"🔍 检查策略: {name}")
                    sig = inspect.signature(obj.__init__)
                    print(f"   构造函数: {sig}")
                    
                    # 测试实例化
                    try:
                        instance = obj({'name': '测试', 'symbols': ['BTC/USDT']})
                        working_strategies.append((name, file_path))
                        print(f"   ✅ 可正常实例化")
                    except Exception as e:
                        broken_strategies.append((name, file_path, str(e)))
                        print(f"   ❌ 实例化失败: {e}")
                    
        except Exception as e:
            print(f"❌ 导入失败 {file_path}: {e}")
    
    print(f"\n📊 审计结果:")
    print(f"   正常策略: {len(working_strategies)}")
    print(f"   异常策略: {len(broken_strategies)}")
    
    return working_strategies, broken_strategies

if __name__ == "__main__":
    working, broken = audit_strategies()