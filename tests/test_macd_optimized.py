# tests/test_macd_optimized.py
import sys
import os
import numpy as np
import asyncio

# 设置路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

from src.strategies.macd_strategy_optimized import MACDStrategyOptimized, TradingSignal, SignalType

def create_better_test_data():
    """创建更好的测试数据 - 包含明显的转折点"""
    prices = []
    current = 50000
    
    # 第一阶段: 明显下跌 (创造MACD负值机会)
    print("📉 创建下跌阶段...")
    for i in range(15):
        current = current * (1 + np.random.normal(-0.003, 0.002))  # 更强下跌
        prices.append(current)
    
    # 第二阶段: 震荡筑底 (创造转折机会)
    print("📊 创建震荡阶段...")
    for i in range(10):
        current = current * (1 + np.random.normal(0.000, 0.003))  # 震荡
        prices.append(current)
    
    # 第三阶段: 强势上涨 (创造买入信号)
    print("📈 创建上涨阶段...")
    for i in range(25):
        current = current * (1 + np.random.normal(0.002, 0.001))  # 强势上涨
        prices.append(current)
    
    return prices

class SimpleMarketData:
    def __init__(self, price, timestamp, symbol="BTC/USDT"):
        self.symbol = symbol
        self.data = [timestamp, price, price+100, price-100, price, 1000]  # OHLCV
        self.timestamp = timestamp

async def test_optimized_macd():
    """测试优化后的MACD策略"""
    print("🧪 测试优化版MACD策略...")
    print("=" * 60)
    
    # 使用更宽松的参数
    strategy = MACDStrategyOptimized(
        name="MACD超宽松版",
        symbols=["BTC/USDT"],
        fast_period=8,           # 更快的反应
        slow_period=21,          # 稍短的慢线
        signal_period=7,         # 更快的信号线
        min_trend_strength=0.0001,  # 大幅降低趋势要求
        hist_threshold=0.00001,     # 更敏感的阈值
        min_trade_interval=1       # 最小间隔
    )
    
    # 使用改进的测试数据
    test_prices = create_better_test_data()
    
    print(f"📊 测试数据: {len(test_prices)} 个价格点")
    print(f"📈 价格范围: {min(test_prices):.2f} - {max(test_prices):.2f}")
    print(f"📉 最大回撤: {(min(test_prices)-test_prices[0])/test_prices[0]*100:.2f}%")
    print(f"📈 总涨幅: {(test_prices[-1]-test_prices[0])/test_prices[0]*100:.2f}%")
    
    signals = []
    
    # 逐步喂数据，模拟实时交易
    for i, price in enumerate(test_prices):
        market_data = SimpleMarketData(price, i)
        signal = await strategy.analyze(market_data)
        
        if signal:
            signals.append(signal)
            print(f"✅ 捕获信号 #{len(signals)}: {signal.signal_type.value} @ {signal.price:.2f}")
            print(f"   原因: {signal.reason}")
            print(f"   强度: {signal.strength:.2f}")
            print("---")
    
    print(f"\n🎉 MACD优化版测试完成")
    print(f"📨 总生成信号: {len(signals)}")
    print(f"📊 测试数据趋势: 开始 {test_prices[0]:.2f} -> 结束 {test_prices[-1]:.2f}")
    
    if signals:
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        print(f"🛒 买入信号: {len(buy_signals)}")
        print(f"🏪 卖出信号: {len(sell_signals)}")
        
        # 简单策略评估
        if len(buy_signals) > 0 and len(sell_signals) > 0:
            first_buy = buy_signals[0].price
            last_sell = sell_signals[-1].price if sell_signals else test_prices[-1]
            profit_pct = (last_sell - first_buy) / first_buy * 100
            print(f"💰 简单收益: {profit_pct:+.2f}%")
    else:
        print("❌ 未生成任何信号，需要进一步调试")
        
        # 诊断信息
        print("\n🔍 诊断信息:")
        print("可能原因:")
        print("1. 趋势强度阈值过高")
        print("2. 柱状图没有负值")
        print("3. MACD金叉条件不满足")
        print("建议:")
        print("1. 进一步降低 min_trend_strength")
        print("2. 调整MACD参数")
        print("3. 检查买入条件逻辑")

if __name__ == "__main__":
    asyncio.run(test_optimized_macd())