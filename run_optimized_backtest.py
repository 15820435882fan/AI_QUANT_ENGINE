# run_optimized_backtest.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化回测测试
"""

import sys
import os
import asyncio
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig, DataManager
from src.backtesting.backtest_strategies import RobustSMAStrategy

async def optimized_backtest():
    """运行优化后的回测"""
    print("🧪 启动优化回测...")
    
    # 测试不同的参数组合
    param_combinations = [
        (5, 20),   # 原始参数
        (10, 30),  # 优化1
        (15, 45),  # 优化2
        (20, 60),  # 优化3
    ]
    
    best_result = None
    best_params = None
    
    for fast, slow in param_combinations:
        print(f"\n🔧 测试参数: 快速MA={fast}, 慢速MA={slow}")
        
        config = BacktestConfig(
            initial_capital=10000.0,
            start_date="2024-01-01", 
            end_date="2024-01-05"  # 缩短测试周期以加快速度
        )
        
        engine = BacktestEngine(config)
        data_manager = DataManager()
        historical_data = await data_manager.load_historical_data(
            "BTC/USDT", config.start_date, config.end_date
        )
        
        strategy = RobustSMAStrategy(
            name=f"SMA_{fast}_{slow}", 
            symbols=["BTC/USDT"],
            fast_period=fast,
            slow_period=slow
        )
        
        result = await engine.run_backtest(strategy, historical_data)
        
        print(f"📊 结果: 收益={result.total_return:.2%}, 交易数={result.total_trades}")
        
        # 选择最佳参数
        if best_result is None or result.total_return > best_result.total_return:
            best_result = result
            best_params = (fast, slow)
    
    print(f"\n🎉 最佳参数组合: 快速MA={best_params[0]}, 慢速MA={best_params[1]}")
    print(f"💰 最佳收益: {best_result.total_return:.2%}")
    print(f"🔢 交易次数: {best_result.total_trades}")
    print(f"📉 最大回撤: {best_result.max_drawdown:.2%}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(optimized_backtest())