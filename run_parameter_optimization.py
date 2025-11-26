# run_parameter_optimization.py
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig, DataManager
from src.backtesting.backtest_strategies import RobustSMAStrategy

async def optimize_parameters():
    """优化策略参数"""
    print("🧪 开始参数优化...")
    
    # 测试不同的参数组合
    param_combinations = [
        # (fast, slow) 组合
        (3, 8),    # 当前 - 太敏感
        (5, 15),   # 优化1
        (8, 21),   # 优化2  
        (10, 30),  # 优化3
        (13, 34),  # 优化4
        (15, 40),  # 优化5
    ]
    
    best_return = -float('inf')
    best_params = None
    
    for fast, slow in param_combinations:
        config = BacktestConfig(initial_capital=10000.0)
        engine = BacktestEngine(config)
        data_manager = DataManager()
        
        historical_data = await data_manager.load_historical_data(
            "BTC/USDT", "2024-01-01", "2024-01-10"
        )
        
        strategy = RobustSMAStrategy(
            name=f"SMA_{fast}_{slow}", 
            symbols=["BTC/USDT"],
            fast_period=fast,
            slow_period=slow
        )
        
        result = await engine.run_backtest(strategy, historical_data)
        
        print(f"🔧 SMA({fast},{slow}): 收益={result.total_return:.2%}, 交易数={result.total_trades}")
        
        if result.total_return > best_return:
            best_return = result.total_return
            best_params = (fast, slow)
    
    print(f"\n🎉 最佳参数: SMA({best_params[0]},{best_params[1]})")
    print(f"💰 最佳收益: {best_return:.2%}")
    return best_params

if __name__ == "__main__":
    asyncio.run(optimize_parameters())