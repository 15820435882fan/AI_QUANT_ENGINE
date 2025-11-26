# test_integration.py
import sys
import os
import asyncio
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.data_pipeline import DataPipeline, DataType
from src.strategies.strategy_orchestrator import StrategyOrchestrator
from src.trading.order_executor import OrderExecutor, Order, OrderType
from src.strategies.simple_moving_average import SimpleMovingAverageStrategy

async def signal_handler(signal, executor):
    """处理交易信号并执行订单"""
    print(f"🚀 执行交易信号: {signal.symbol} {signal.signal_type.value} 价格: {signal.price:.2f}")
    
    # 风险检查
    if not await executor.risk_check(signal):
        print("⛔ 交易被风险控制阻止")
        return
    
    # 创建订单
    order = Order(
        symbol=signal.symbol,
        order_type=OrderType.MARKET,
        side=signal.signal_type.value,
        amount=0.01,  # 增加到0.01以便观察资金变化
        price=signal.price
    )
    
    # 执行订单
    order_id = await executor.create_order(order)
    if order_id:
        print(f"✅ 订单执行成功: {order_id}")
        # 更新持仓
        quantity = order.amount if signal.signal_type.value == 'buy' else -order.amount
        executor.update_position(signal.symbol, quantity, signal.price)
        return order_id
    return None

async def test_full_system():
    """测试完整的量化交易系统"""
    print("🧪 开始完整系统测试...")
    
    # 1. 创建所有组件
    pipeline = DataPipeline(symbols=["BTC/USDT"])
    orchestrator = StrategyOrchestrator()
    executor = OrderExecutor()
    
    # 2. 创建并注册策略
    sma_strategy = SimpleMovingAverageStrategy(
        name="SMA策略", 
        symbols=["BTC/USDT"],
        fast_period=5,
        slow_period=10
    )
    orchestrator.register_strategy(sma_strategy)
    
    # 3. 初始化所有组件
    await pipeline.initialize()
    await executor.initialize()
    
    # 4. 连接所有模块 - 修复重复订阅问题
    async def handle_market_data(market_data):
        """统一处理市场数据"""
        if market_data.data_type == DataType.OHLCV:
            print(f"📈 收到K线数据，开始策略分析...")
            # 只有OHLCV数据才生成交易信号
            signals = await orchestrator.generate_signals(market_data)
            print(f"📋 生成 {len(signals)} 个交易信号")
            for signal in signals:
                order_id = await signal_handler(signal, executor)
                if order_id:
                    print(f"🎉 成功执行订单: {order_id}")
        else:
            # TICKER数据只打印日志
            print(f"📊 收到行情数据: {market_data.symbol} 价格: {market_data.data.get('last', 'N/A')}")
    
    # 只订阅一次，统一处理
    pipeline.subscribe(DataType.TICKER, handle_market_data)
    pipeline.subscribe(DataType.OHLCV, handle_market_data)
    
    # 5. 启动系统
    await pipeline.start()
    
    print("🚀 全自动交易系统运行中... 等待300秒")
    print("💡 系统将自动分析数据、生成信号、执行交易!")
    await asyncio.sleep(300) # 延长到5分钟，确保收集足够K线数据
    
    # 6. 停止系统并显示结果
    await pipeline.stop()
    
    # 显示交易结果
    print("\n📊 交易结果汇总:")
    print(f"💰 最终资金: {executor.balance:.2f} USDT")
    print(f"📈 持仓情况: {executor.positions}")
    print(f"📋 总订单数: {len(executor.orders)}")
    
    print("✅ 完整系统测试完成！")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_full_system())