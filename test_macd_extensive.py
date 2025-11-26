# test_macd_extensive.py
#!/usr/bin/env python3
import sys
import os
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 设置项目根目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.strategies.macd_strategy_smart import MACDStrategySmart, SignalType

def create_extensive_test_data(days=180, volatility=0.02):
    """创建更长时间的测试数据，模拟真实市场"""
    print(f"📊 创建 {days} 天测试数据...")
    
    prices = []
    current = 50000  # 起始价格
    
    # 模拟真实市场的不同阶段
    phases = [
        ("📈 牛市阶段", 30, 0.003, 0.015),   # 30天，平均上涨0.3%
        ("📉 回调阶段", 10, -0.002, 0.020), # 10天，回调
        ("📊 震荡阶段", 40, 0.000, 0.025),  # 40天，横盘
        ("🚀 突破阶段", 20, 0.004, 0.018),  # 20天，强势上涨
        ("🐻 熊市阶段", 30, -0.002, 0.022), # 30天，下跌
        ("🔄 复苏阶段", 50, 0.001, 0.016),  # 50天，缓慢复苏
    ]
    
    total_days = 0
    for phase_name, phase_days, trend, phase_vol in phases:
        if total_days >= days:
            break
            
        print(f"   {phase_name}: {phase_days}天")
        for i in range(phase_days):
            if total_days >= days:
                break
                
            # 添加一些市场噪音和趋势
            daily_change = np.random.normal(trend, phase_vol)
            current = current * (1 + daily_change)
            prices.append(current)
            total_days += 1
    
    return prices

async def test_extensive_macd():
    """全面测试MACD智能策略"""
    print("🧪 开始MACD策略全面回测...")
    print("=" * 60)
    
    strategy = MACDStrategySmart(
        name="MACD全面测试",
        symbols=["BTC/USDT"],
        fast_period=12,
        slow_period=26,
        signal_period=9,
        min_trade_interval=5
    )
    
    # 创建更长时间的测试数据
    test_prices = create_extensive_test_data(days=180)  # 6个月数据
    
    print(f"📊 测试数据: {len(test_prices)} 个价格点")
    print(f"📈 价格范围: {min(test_prices):.2f} - {max(test_prices):.2f}")
    print(f"📉 最大回撤: {(min(test_prices)-test_prices[0])/test_prices[0]*100:.2f}%")
    print(f"📈 总涨幅: {(test_prices[-1]-test_prices[0])/test_prices[0]*100:.2f}%")
    
    class SimpleMarketData:
        def __init__(self, price, timestamp):
            self.symbol = "BTC/USDT"
            self.data = [timestamp, price, price+50, price-50, price, 1000]
            self.timestamp = timestamp
    
    signals = []
    
    print("🔄 开始回测...")
    for i, price in enumerate(test_prices):
        market_data = SimpleMarketData(price, i)
        signal = await strategy.analyze(market_data)
        
        if signal:
            signals.append(signal)
            print(f"✅ 信号 #{len(signals)}: {signal.signal_type.value} @ {signal.price:.2f}")
    
    # 详细分析结果
    print(f"\n🎉 全面回测完成")
    print("=" * 50)
    print(f"📨 总交易信号: {len(signals)}")
    
    if signals:
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        
        print(f"🛒 买入信号: {len(buy_signals)}")
        print(f"🏪 卖出信号: {len(sell_signals)}")
        
        # 计算交易对收益
        total_profit = 0
        profitable_trades = 0
        trade_details = []
        
        for i in range(min(len(buy_signals), len(sell_signals))):
            buy_price = buy_signals[i].price
            sell_price = sell_signals[i].price
            profit_pct = (sell_price - buy_price) / buy_price * 100
            total_profit += profit_pct
            
            if profit_pct > 0:
                profitable_trades += 1
            
            trade_details.append({
                'trade': i+1,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'profit_pct': profit_pct,
                'profitable': profit_pct > 0
            })
        
        if trade_details:
            # 显示前5笔交易详情
            print(f"\n📊 交易详情 (前5笔):")
            for trade in trade_details[:5]:
                status = "✅ 盈利" if trade['profitable'] else "❌ 亏损"
                print(f"   交易 {trade['trade']}: {trade['buy_price']:.2f} → {trade['sell_price']:.2f} = {trade['profit_pct']:+.2f}% {status}")
            
            # 总体统计
            avg_profit = total_profit / len(trade_details)
            win_rate = profitable_trades / len(trade_details) * 100
            
            print(f"\n📈 总体表现:")
            print(f"   总交易次数: {len(trade_details)}")
            print(f"   盈利交易: {profitable_trades}")
            print(f"   胜率: {win_rate:.1f}%")
            print(f"   平均收益: {avg_profit:+.2f}%")
            print(f"   总收益: {total_profit:+.2f}%")
            
            # 风险评估
            profits = [t['profit_pct'] for t in trade_details]
            max_drawdown = min(profits) if profits else 0
            profit_std = np.std(profits) if len(profits) > 1 else 0
            
            print(f"🔍 风险评估:")
            print(f"   最大单笔亏损: {max_drawdown:+.2f}%")
            print(f"   收益波动率: {profit_std:.2f}%")
            
            # 简单夏普比率（假设无风险利率为0）
            sharpe_ratio = avg_profit / profit_std if profit_std > 0 else 0
            print(f"   夏普比率: {sharpe_ratio:.2f}")
            
            # 策略评价
            if win_rate > 60 and avg_profit > 1:
                print("🎯 策略评价: ✅ 优秀")
            elif win_rate > 50 and avg_profit > 0:
                print("🎯 策略评价: ⚡ 良好") 
            else:
                print("🎯 策略评价: 🔧 需要优化")

if __name__ == "__main__":
    asyncio.run(test_extensive_macd())