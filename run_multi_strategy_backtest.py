# run_multi_strategy_backtest.py
#!/usr/bin/env python3
import sys
import os
import asyncio
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig, DataManager
from src.backtesting.backtest_strategies import RobustSMAStrategy
from src.strategies.macd_strategy import MACDStrategy

async def compare_strategies():
    """对比多个策略表现"""
    print("🧪 开始多策略对比回测...")
    print("=" * 60)
    
    strategies = [
        ("SMA策略", RobustSMAStrategy, {"fast_period": 10, "slow_period": 30}),
        ("MACD策略", MACDStrategy, {"fast_period": 12, "slow_period": 26, "signal_period": 9}),
    ]
    
    results = {}
    
    for strategy_name, strategy_class, params in strategies:
        print(f"\n📊 测试策略: {strategy_name}")
        
        config = BacktestConfig(
            initial_capital=10000.0,
            start_date="2024-01-01",
            end_date="2024-01-10"
        )
        
        engine = BacktestEngine(config)
        data_manager = DataManager()
        
        historical_data = await data_manager.load_historical_data(
            "BTC/USDT", config.start_date, config.end_date
        )
        
        # 创建策略实例
        strategy = strategy_class(
            name=strategy_name,
            symbols=["BTC/USDT"],
            **params
        )
        
        result = await engine.run_backtest(strategy, historical_data)
        results[strategy_name] = result
        
        print(f"💰 收益: {result.total_return:.2%}")
        print(f"🔢 交易次数: {result.total_trades}")
        print(f"📉 最大回撤: {result.max_drawdown:.2%}")
    
    # 找出最佳策略
    best_strategy = max(results.items(), key=lambda x: x[1].total_return)
    
    print(f"\n🎉 最佳策略: {best_strategy[0]}")
    print(f"📈 最佳收益: {best_strategy[1].total_return:.2%}")
    print(f"⚡ 夏普比率: {best_strategy[1].sharpe_ratio:.2f}")
    
    return results

async def optimize_macd_parameters():
    """优化MACD参数"""
    print("\n🔧 开始MACD参数优化...")
    
    param_combinations = [
        (8, 21, 5),   # 快速
        (12, 26, 9),  # 标准
        (5, 35, 5),   # 宽幅
        (6, 19, 9),   # 敏感
    ]
    
    best_return = -float('inf')
    best_params = None
    
    for fast, slow, signal in param_combinations:
        config = BacktestConfig(initial_capital=10000.0)
        engine = BacktestEngine(config)
        data_manager = DataManager()
        
        historical_data = await data_manager.load_historical_data(
            "BTC/USDT", "2024-01-01", "2024-01-10"
        )
        
        strategy = MACDStrategy(
            name=f"MACD_{fast}_{slow}_{signal}",
            symbols=["BTC/USDT"],
            fast_period=fast,
            slow_period=slow,
            signal_period=signal
        )
        
        result = await engine.run_backtest(strategy, historical_data)
        
        print(f"MACD({fast},{slow},{signal}): {result.total_return:.2%}")
        
        if result.total_return > best_return:
            best_return = result.total_return
            best_params = (fast, slow, signal)
    
    print(f"\n🎯 最佳MACD参数: {best_params}")
    print(f"💰 最佳收益: {best_return:.2%}")
    
    return best_params

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        # 1. 对比策略
        await compare_strategies()
        
        # 2. 优化参数
        await optimize_macd_parameters()
    
    asyncio.run(main())