# scripts/test_integration.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成测试 - 状态机 + 数据管道
"""

import asyncio
import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.state_controller import StateController, TradingState
from data.data_pipeline import DataPipeline, DataType, MarketData

class IntegratedEngine:
    """集成测试引擎"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.state_controller = StateController(self)
        self.data_pipeline = None
        self.received_data_count = 0
    
    async def start_integrated_system(self):
        """启动集成系统"""
        self.logger.info("🚀 启动集成系统...")
        
        # 1. 启动状态机
        await self.state_controller.start()
        
        # 2. 创建并启动数据管道
        self.data_pipeline = DataPipeline(symbols=["BTC/USDT"])
        
        # 订阅数据更新
        self.data_pipeline.subscribe(DataType.TICKER, self.handle_market_data)
        self.data_pipeline.subscribe(DataType.OHLCV, self.handle_market_data)
        
        await self.data_pipeline.initialize()
        await self.data_pipeline.start()
        
        self.logger.info("✅ 集成系统启动完成")
    
    async def handle_market_data(self, data: MarketData):
        """处理市场数据"""
        self.received_data_count += 1
        self.logger.info(f"📨 收到市场数据 #{self.received_data_count}: {data.symbol} {data.data_type}")
        
        # 当收到数据时，通知状态机
        if self.received_data_count == 1:
            await self.state_controller.handle_event("data_ready")
    
    async def stop_system(self):
        """停止系统"""
        if self.data_pipeline:
            await self.data_pipeline.stop()
        self.logger.info("🛑 集成系统已停止")

async def test_integration():
    """测试集成系统"""
    print("🧪 测试状态机 + 数据管道集成...")
    
    engine = IntegratedEngine()
    
    try:
        # 启动集成系统
        await engine.start_integrated_system()
        
        # 运行一段时间观察交互
        print("⏳ 观察系统运行...")
        await asyncio.sleep(15)
        
        print(f"📊 最终状态: {engine.state_controller.current_state}")
        print(f"📨 总共接收数据: {engine.received_data_count} 条")
        
    finally:
        await engine.stop_system()
    
    print("✅ 集成测试完成!")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 50)
    print("🔗 状态机 + 数据管道集成测试")
    print("=" * 50)
    
    asyncio.run(test_integration())