# test_multi_strategy_comprehensive.py
#!/usr/bin/env python3
import sys
import os
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import List  # 添加这行

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from multi_strategy_manager_enhanced import EnhancedMultiStrategyManager
from src.strategies.strategy_orchestrator import TradingSignal, SignalType

def create_realistic_market_data(days=180, base_price=50000):
    """创建更真实的市场数据，包含趋势、震荡、突破等各种模式"""
    print("📊 生成真实市场数据...")
    
    prices = []
    current = base_price
    timestamps = []
    
    # 定义市场阶段
    phases = [
        ("📈 缓慢上涨", 15, 0.001, 0.008),    # 缓慢牛市
        ("📊 横盘震荡", 20, 0.000, 0.012),    # 震荡市 - 布林带策略的机会
        ("🚀 强势突破", 10, 0.003, 0.010),    # 趋势市 - 海龟策略的机会
        ("📉 深度回调", 15, -0.002, 0.015),   # 下跌市
        ("🔄 震荡反弹", 20, 0.001, 0.018),    # 高波动震荡
        ("🎯 趋势确立", 10, 0.002, 0.009),    # 趋势市 - MACD策略的机会
    ]
    
    current_time = datetime.now() - timedelta(days=days)
    day_count = 0
    
    for phase_name, phase_days, trend, volatility in phases:
        print(f"   {phase_name}: {phase_days}天")
        
        for day in range(phase_days):
            if day_count >= days:
                break
                
            # 生成日内的4个价格点（模拟小时数据）
            for hour in [9, 12, 15, 18]:  # 一天4个时间点
                # 添加趋势和随机波动
                change = np.random.normal(trend/4, volatility/2)
                current = current * (1 + change)
                prices.append(current)
                
                timestamp = current_time + timedelta(days=day_count, hours=hour)
                timestamps.append(timestamp.timestamp())
            
            day_count += 1
            current_time += timedelta(days=1)
    
    return prices, timestamps

class TestMarketData:
    """测试用市场数据"""
    def __init__(self, price, high, low, timestamp, symbol="BTC/USDT"):
        self.symbol = symbol
        self.data = [timestamp, price, high, low, price, 1000]  # OHLCV格式
        self.timestamp = timestamp
        self.close = price

async def analyze_multi_strategy_performance(signals: List[TradingSignal], prices: List[float], strategy_manager):
    """分析多策略组合性能"""
    print("📈 多策略组合性能分析")
    print("=" * 40)
    
    if not signals:
        print("❌ 没有生成任何交易信号")
        return
    
    # 基本统计
    buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
    sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
    
    print(f"📨 总交易信号: {len(signals)}")
    print(f"🛒 买入信号: {len(buy_signals)}")
    print(f"🏪 卖出信号: {len(sell_signals)}")
    print(f"📊 交易频率: {len(signals)/len(prices)*100:.2f}%")
    
    # 策略性能统计
    strategy_stats = strategy_manager.get_strategy_performance()
    print(f"\n🔧 各策略贡献:")
    for strategy, stats in strategy_stats.items():
        print(f"   {strategy}: {stats['signal_count']} 个信号")
    
    # 交易对分析
    if buy_signals and sell_signals:
        total_profit = 0
        profitable_trades = 0
        trade_details = []
        
        # 配对交易（简单按顺序配对）
        pairs = min(len(buy_signals), len(sell_signals))
        for i in range(pairs):
            buy_signal = buy_signals[i]
            sell_signal = sell_signals[i]
            
            # 确保卖出在买入之后
            if sell_signal.timestamp > buy_signal.timestamp:
                profit_pct = (sell_signal.price - buy_signal.price) / buy_signal.price * 100
                total_profit += profit_pct
                
                if profit_pct > 0:
                    profitable_trades += 1
                
                trade_details.append({
                    'pair': i+1,
                    'buy_price': buy_signal.price,
                    'sell_price': sell_signal.price,
                    'profit_pct': profit_pct,
                    'hold_days': (sell_signal.timestamp - buy_signal.timestamp) / (24*3600),
                    'profitable': profit_pct > 0
                })
        
        if trade_details:
            # 显示交易详情
            print(f"\n📊 交易对详情 (前5笔):")
            for trade in trade_details[:5]:
                status = "✅ 盈利" if trade['profitable'] else "❌ 亏损"
                print(f"   交易 {trade['pair']}: {trade['buy_price']:.2f} → {trade['sell_price']:.2f}")
                print(f"       收益: {trade['profit_pct']:+.2f}% | 持仓: {trade['hold_days']:.1f}天 {status}")
            
            # 总体统计
            avg_profit = total_profit / len(trade_details)
            win_rate = profitable_trades / len(trade_details) * 100
            avg_hold_days = sum(t['hold_days'] for t in trade_details) / len(trade_details)
            
            print(f"\n📈 总体表现:")
            print(f"   总交易次数: {len(trade_details)}")
            print(f"   盈利交易: {profitable_trades}")
            print(f"   胜率: {win_rate:.1f}%")
            print(f"   平均收益: {avg_profit:+.2f}%")
            print(f"   总收益: {total_profit:+.2f}%")
            print(f"   平均持仓: {avg_hold_days:.1f}天")
            
            # 风险评估
            profits = [t['profit_pct'] for t in trade_details]
            max_drawdown = min(profits)
            profit_std = np.std(profits) if len(profits) > 1 else 0
            
            print(f"🔍 风险评估:")
            print(f"   最大单笔亏损: {max_drawdown:+.2f}%")
            print(f"   收益波动率: {profit_std:.2f}%")
            
            # 夏普比率（年化，假设无风险利率0）
            sharpe_ratio = (avg_profit / profit_std) * np.sqrt(365/avg_hold_days) if profit_std > 0 else 0
            print(f"   年化夏普比率: {sharpe_ratio:.2f}")
            
            # 策略评价
            if win_rate > 60 and avg_profit > 2 and sharpe_ratio > 1:
                print("🎯 策略评价: 🏆 优秀")
            elif win_rate > 50 and avg_profit > 0 and sharpe_ratio > 0.5:
                print("🎯 策略评价: ✅ 良好")
            else:
                print("🎯 策略评价: 🔧 需要优化")

async def test_multi_strategy_comprehensive():
    """全面测试多策略组合"""
    print("🧪 开始多策略组合全面测试")
    print("=" * 60)
    
    # 初始化多策略管理器
    strategy_manager = EnhancedMultiStrategyManager(symbols=["BTC/USDT"])
    
    # 创建测试数据
    test_prices, test_timestamps = create_realistic_market_data(days=90, base_price=50000)
    
    print(f"📊 测试数据: {len(test_prices)} 个价格点 (约{len(test_prices)//4}个交易日)")
    print(f"📈 价格范围: {min(test_prices):.2f} - {max(test_prices):.2f}")
    print(f"📉 最大回撤: {(min(test_prices)-test_prices[0])/test_prices[0]*100:.2f}%")
    print(f"📈 总涨幅: {(test_prices[-1]-test_prices[0])/test_prices[0]*100:.2f}%")
    
    # 模拟市场状态变化（简化版）
    market_regimes = [
        (0, "bull"),      # 开始阶段：牛市
        (len(test_prices)//3, "ranging"),  # 1/3处：震荡市
        (2*len(test_prices)//3, "trend")   # 2/3处：趋势市
    ]
    
    signals = []
    regime_index = 0
    
    print("\n🔄 开始多策略回测...")
    
    for i, (price, timestamp) in enumerate(zip(test_prices, test_timestamps)):
        # 更新市场状态
        if regime_index < len(market_regimes) and i >= market_regimes[regime_index][0]:
            strategy_manager.update_market_regime(market_regimes[regime_index][1])
            regime_index += 1
        
        # 生成高低价（简化处理）
        high_price = price * (1 + abs(np.random.normal(0, 0.005)))
        low_price = price * (1 - abs(np.random.normal(0, 0.005)))
        
        # 创建市场数据
        market_data = TestMarketData(price, high_price, low_price, timestamp)
        
        # 多策略分析
        signal = await strategy_manager.analyze(market_data)
        
        if signal:
            signals.append(signal)
            print(f"✅ 最终信号 #{len(signals)}: {signal.signal_type.value} @ {signal.price:.2f}")
            print(f"   原因: {signal.reason}")
            print("---")
    
    # 详细性能分析
    print(f"\n🎉 多策略回测完成")
    print("=" * 50)
    await analyze_multi_strategy_performance(signals, test_prices, strategy_manager)

if __name__ == "__main__":
    asyncio.run(test_multi_strategy_comprehensive())