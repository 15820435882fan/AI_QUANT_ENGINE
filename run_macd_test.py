# run_macd_test.py
#!/usr/bin/env python3
import sys
import os

# 设置项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from tests.test_macd_optimized import test_optimized_macd
import asyncio

if __name__ == "__main__":
    asyncio.run(test_optimized_macd())