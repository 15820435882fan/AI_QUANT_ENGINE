# scripts/test_state_machine.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试状态机功能
"""

import asyncio
import logging
import sys
import os

# 添加src到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.state_controller import StateController, TradingState

class TestEngine:
    """测试用引擎模拟类"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)

async def test_full_state_flow():
    """测试完整状态流转"""
    print("🧪 测试完整状态流转...")
    
    engine = TestEngine()
    controller = StateController(engine)
    
    # 启动状态机
    await controller.start()
    
    print("✅ 状态机测试完成!")
    print(f"📊 最终状态: {controller.current_state}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 50)
    print("🎯 状态控制器测试")
    print("=" * 50)
    
    asyncio.run(test_full_state_flow())