# src/core/complete_engine.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整交易引擎 - 整合所有组件
"""

import asyncio
import logging
import sys
import os
from typing import List, Dict, Any

# 添加src目录到Python路径，确保可以找到其他模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 使用绝对导入
from core.state_controller import StateController, TradingState
from data.data_pipeline import DataPipeline, DataType, MarketData
from strategies.strategy_orchestrator import StrategyOrchestrator, MovingAverageStrategy, RSIStrategy, SignalType, TradingSignal
from trading.order_executor import MockOrderExecutor, Order, OrderType, OrderStatus

class CompleteTradingEngine:
    """
    完整交易引擎 - 整合所有组件的工作引擎
    """
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ["BTC/USDT"]
        self.logger = logging.getLogger(__name__)
        
        # 初始化所有组件
        self.state_controller = StateController(self)
        self.data_pipeline = DataPipeline(symbols=self.symbols)
        self.strategy_orchestrator = StrategyOrchestrator()
        self.order_executor = MockOrderExecutor()  # 使用模拟执行器
        
        # 注册策略
        self.strategy_orchestrator.register_strategy(MovingAverageStrategy())
        self.strategy_orchestrator.register_strategy(RSIStrategy())
        
        # 状态变量
        self.received_data_count = 0
        self.generated_signals = []
        self.executed_orders = []
        self.is_running = False
    
    async def start(self):
        """启动完整交易引擎"""
        self.logger.info("🚀 启动完整交易引擎...")
        self.is_running = True
        
        try:
            # 1. 初始化订单执行器
            await self.order_executor.initialize()
            
            # 2. 启动状态机
            await self.state_controller.start()
            
            # 3. 启动数据管道并订阅
            self.data_pipeline.subscribe(DataType.TICKER, self._handle_market_data)
            self.data_pipeline.subscribe(DataType.OHLCV, self._handle_market_data)
            await self.data_pipeline.initialize()
            await self.data_pipeline.start()
            
            self.logger.info("✅ 完整交易引擎启动完成")
            
            # 4. 运行主循环
            await self._run_trading_loop()
            
        except Exception as e:
            self.logger.error(f"❌ 引擎启动失败: {e}")
            await self.stop()
    
    async def _run_trading_loop(self):
        """运行交易主循环"""
        self.logger.info("🔄 进入交易主循环...")
        
        while self.is_running:
            try:
                # 检查并处理订单状态
                await self._monitor_orders()
                
                # 短暂休眠，避免过度占用CPU
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"❌ 交易循环错误: {e}")
                await asyncio.sleep(5)  # 出错后等待更久
    
    async def _handle_market_data(self, market_data: MarketData):
        """处理市场数据"""
        self.received_data_count += 1
        self.logger.info(f"📨 收到市场数据 #{self.received_data_count}: {market_data.symbol}")
        
        # 使用策略引擎分析数据
        signal = await self.strategy_orchestrator.analyze_market(market_data)
        
        if signal:
            self.generated_signals.append(signal)
            self.logger.info(f"🎯 策略信号: {signal.signal_type.value} - {signal.reason}")
            
            # 根据信号执行交易
            await self._execute_trading_signal(signal)
        
        # 通知状态机数据就绪
        if self.received_data_count >= 2:  # 收到足够数据后开始分析
            await self.state_controller.handle_event("data_ready")
    
    async def _execute_trading_signal(self, signal: TradingSignal):
        """执行交易信号"""
        self.logger.info(f"💼 执行交易信号: {signal.signal_type.value}")
        
        # 创建订单
        order = Order(
            symbol=signal.symbol,
            order_type=OrderType.MARKET,
            side=signal.signal_type.value,  # 'buy' or 'sell'
            amount=0.001,  # 固定数量，实际中应该根据资金管理计算
            price=signal.price
        )
        
        # 提交订单
        order_id = await self.order_executor.create_order(order)
        if order_id:
            self.executed_orders.append(order)
            self.logger.info(f"✅ 订单提交成功: {order_id}")
            
            # 通知状态机进入监控状态
            await self.state_controller.handle_event("order_placed")
        else:
            self.logger.error("❌ 订单提交失败")
    
    async def _monitor_orders(self):
        """监控订单状态"""
        open_orders = await self.order_executor.get_open_orders()
        if open_orders:
            self.logger.info(f"👀 监控 {len(open_orders)} 个未成交订单")
            
            # 检查订单是否完全成交
            for order in open_orders:
                if order.status == OrderStatus.CLOSED:
                    self.logger.info(f"✅ 订单完全成交: {order.order_id}")
                    # 可以在这里触发下一步操作
    
    async def stop(self):
        """停止交易引擎"""
        self.logger.info("🛑 停止完整交易引擎...")
        self.is_running = False
        
        if self.data_pipeline:
            await self.data_pipeline.stop()
        
        self.logger.info("✅ 交易引擎已停止")
    
    def get_status_report(self) -> Dict[str, Any]:
        """获取状态报告"""
        return {
            "running": self.is_running,
            "data_received": self.received_data_count,
            "signals_generated": len(self.generated_signals),
            "orders_executed": len(self.executed_orders),
            "current_state": self.state_controller.current_state.value if self.state_controller.current_state else None
        }

# 测试完整引擎
async def test_complete_engine():
    """测试完整交易引擎"""
    print("🧪 测试完整交易引擎...")
    
    engine = CompleteTradingEngine(symbols=["BTC/USDT"])
    
    try:
        # 启动引擎
        await engine.start()
        
        # 运行一段时间
        print("⏳ 完整引擎运行中...")
        await asyncio.sleep(30)  # 运行更长时间以收集足够数据
        
        # 输出状态报告
        report = engine.get_status_report()
        print("\n📊 完整引擎状态报告:")
        for key, value in report.items():
            print(f"  {key}: {value}")
        
        # 显示交易信号和订单
        print(f"\n🎯 生成的交易信号: {len(engine.generated_signals)} 个")
        for i, signal in enumerate(engine.generated_signals):
            print(f"  {i+1}. {signal.signal_type.value} - {signal.reason}")
        
        print(f"\n💳 执行的订单: {len(engine.executed_orders)} 个")
        for i, order in enumerate(engine.executed_orders):
            print(f"  {i+1}. {order.side} {order.amount} {order.symbol} - {order.status.value}")
        
    finally:
        await engine.stop()
    
    print("✅ 完整引擎测试完成!")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("🤖 完整AI量化交易引擎测试")
    print("=" * 60)
    
    asyncio.run(test_complete_engine())