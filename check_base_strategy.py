# check_base_strategy.py
#!/usr/bin/env python3
import sys
import os

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from src.strategies.strategy_orchestrator import BaseStrategy
    import inspect
    
    print("🔍 检查BaseStrategy构造函数:")
    sig = inspect.signature(BaseStrategy.__init__)
    print(f"BaseStrategy.__init__ 参数: {sig}")
    
except Exception as e:
    print(f"❌ 检查失败: {e}")