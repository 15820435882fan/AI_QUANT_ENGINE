# test_macd_smart.py
#!/usr/bin/env python3
import sys
import os
import asyncio

# 设置项目根目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入策略
from src.strategies.macd_strategy_smart import test_smart_macd

if __name__ == "__main__":
    print("🚀 启动MACD智能策略测试...")
    asyncio.run(test_smart_macd())