# scripts/test_with_strategies.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成测试 - 加入策略系统
"""

import asyncio
import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.state_controller import StateController, TradingState
from data.data_pipeline import DataPipeline, DataType, MarketData
from strategies.strategy_orchestrator import StrategyOrchestrator, MovingAverageStrategy, RSIStrategy

class AdvancedEngine:
    """高级引擎 - 包含策略系统"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.state_controller = StateController(self)
        self.data_pipeline = None
        self.strategy_orchestrator = StrategyOrchestrator()
        self.received_data_count = 0
        self.trading_signals = []
        
        # 注册策略
        self.strategy_orchestrator.register_strategy(MovingAverageStrategy())
        self.strategy_orchestrator.register_strategy(RSIStrategy())
    
    async def start_advanced_system(self):
        """启动高级系统"""
        self.logger.info("🚀 启动高级交易系统...")
        
        # 启动状态机
        await self.state_controller.start()
        
        # 启动数据管道
        self.data_pipeline = DataPipeline(symbols=["BTC/USDT"])
        self.data_pipeline.subscribe(DataType.TICKER, self.handle_market_data)
        self.data_pipeline.subscribe(DataType.OHLCV, self.handle_market_data)
        
        await self.data_pipeline.initialize()
        await self.data_pipeline.start()
        
        self.logger.info("✅ 高级系统启动完成")
    
    async def handle_market_data(self, data: MarketData):
        """处理市场数据"""
        self.received_data_count += 1
        self.logger.info(f"📨 收到市场数据 #{self.received_data_count}: {data.symbol}")
        
        # 使用策略系统分析数据
        signal = await self.strategy_orchestrator.analyze_market(data)
        if signal:
            self.trading_signals.append(signal)
            self.logger.info(f"🎯 策略信号: {signal.signal_type.value} - {signal.reason}")
            
            # 通知状态机（简化逻辑）
            if self.received_data_count == 2:  # 收到足够数据后开始分析
                await self.state_controller.handle_event("data_ready")
    
    async def stop_system(self):
        """停止系统"""
        if self.data_pipeline:
            await self.data_pipeline.stop()
        self.logger.info("🛑 高级系统已停止")

async def test_advanced_system():
    """测试高级系统"""
    print("🧪 测试完整交易系统（含策略引擎）...")
    
    engine = AdvancedEngine()
    
    try:
        await engine.start_advanced_system()
        
        # 运行更长时间以观察策略信号
        print("⏳ 观察策略系统运行...")
        await asyncio.sleep(25)
        
        print(f"📊 最终状态: {engine.state_controller.current_state}")
        print(f"📨 总共接收数据: {engine.received_data_count} 条")
        print(f"🎯 生成交易信号: {len(engine.trading_signals)} 个")
        
        # 显示所有信号
        for i, signal in enumerate(engine.trading_signals):
            print(f"  {i+1}. {signal.signal_type.value} - {signal.reason}")
        
    finally:
        await engine.stop_system()
    
    print("✅ 高级系统测试完成!")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("🤖 完整交易系统测试（状态机 + 数据管道 + 策略引擎）")
    print("=" * 60)
    
    asyncio.run(test_advanced_system())