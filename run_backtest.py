# run_backtest.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测系统启动器
"""

import sys
import os
import asyncio
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 🔧 添加必要的导入
from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig, DataManager
from src.backtesting.backtest_strategies import RobustSMAStrategy

async def main():
    """运行回测系统"""
    print("🧪 启动回测系统...")
    
    # 创建回测配置
    config = BacktestConfig(
        initial_capital=10000.0,
        start_date="2024-01-01", 
        end_date="2024-01-10"
    )
    
    # 创建回测引擎
    engine = BacktestEngine(config)
    
    # 加载历史数据
    data_manager = DataManager()
    historical_data = await data_manager.load_historical_data(
        "BTC/USDT", config.start_date, config.end_date
    )
    
    print(f"📊 加载了 {len(historical_data)} 条历史数据")
    
    # 🔧 使用鲁棒的回测专用策略
    strategy = RobustSMAStrategy(
        name="优化SMA策略", 
        symbols=["BTC/USDT"],
        fast_period=10,
        slow_period=30,
    )
    
    # 运行回测
    print("🚀 开始回测...")
    result = await engine.run_backtest(strategy, historical_data)
    
    # 显示结果
    print(f"\n🎉 回测完成!")
    print(f"📊 回测结果汇总:")
    print(f"💰 总收益: {result.total_return:.2%}")
    print(f"📈 年化收益: {result.annual_return:.2%}") 
    print(f"⚡ 夏普比率: {result.sharpe_ratio:.2f}")
    print(f"📉 最大回撤: {result.max_drawdown:.2%}")
    print(f"🎯 胜率: {result.win_rate:.2%}")
    print(f"🔢 总交易次数: {result.total_trades}")
    print(f"💰 最终资金: {result.final_balance:.2f} USDT")
    
    if result.total_trades > 0:
        print("✅ 策略产生了交易信号!")
        # 显示交易详情
        print(f"\n📋 交易记录:")
        for i, trade in enumerate(result.trades[:5]):  # 显示前5笔交易
            print(f"  {i+1}. {trade['timestamp']} {trade['signal_type']} {trade['quantity']:.4f} @ {trade['price']:.2f}")
        if len(result.trades) > 5:
            print(f"  ... 还有 {len(result.trades) - 5} 笔交易")
    else:
        print("💡 策略未产生交易，可能需要调整参数或延长测试周期")
    
    print("\n✅ 回测系统测试完成")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())