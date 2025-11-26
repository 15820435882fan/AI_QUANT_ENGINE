# test_compound_engine.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from adaptive_compound_engine import AdaptiveCompoundEngine
from src.strategies.trend_following_compound import TrendFollowingCompound
from src.strategies.mean_reversion_compound import MeanReversionCompound

def generate_test_data(periods: int = 200) -> pd.DataFrame:
    """生成测试数据"""
    np.random.seed(42)
    
    # 生成更真实的价格序列（包含趋势和震荡）
    prices = [100]
    trend = 0.001
    volatility = 0.02
    
    for i in range(1, periods):
        # 模拟市场状态变化
        if i % 50 == 0:  # 每50个周期改变趋势
            trend = np.random.choice([-0.002, 0, 0.002])
            volatility = np.random.uniform(0.01, 0.03)
        
        change = np.random.normal(trend, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 10))  # 防止价格归零
    
    dates = [datetime.now() - timedelta(minutes=5*i) for i in range(periods)][::-1]
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000, 100000) for _ in prices]
    })
    
    data.set_index('timestamp', inplace=True)
    return data

def test_compound_engine():
    """测试复利引擎"""
    print("🚀 测试自适应复利引擎...")
    
    # 创建引擎
    engine = AdaptiveCompoundEngine(initial_capital=10000.0)
    
    # 添加策略
    trend_strategy = TrendFollowingCompound({
        'name': '趋势跟踪',
        'weight': 0.6,
        'parameters': {'fast_window': 10, 'slow_window': 30}
    })
    
    mean_reversion_strategy = MeanReversionCompound({
        'name': '均值回归', 
        'weight': 0.4,
        'parameters': {'bb_period': 20, 'bb_std': 2.0}
    })
    
    engine.add_strategy(trend_strategy)
    engine.add_strategy(mean_reversion_strategy)
    
    # 生成测试数据
    test_data = generate_test_data(100)
    print(f"📊 测试数据: {len(test_data)} 条记录")
    
    # 测试信号生成
    compound_signals = engine.generate_compound_signals(test_data)
    
    print(f"\n🎯 复合信号结果:")
    print(f"  市场状态: {compound_signals['market_regime']}")
    print(f"  最终信号: {compound_signals['final_signal']:.3f}")
    print(f"  综合置信度: {compound_signals['combined_confidence']:.2f}")
    print(f"  交易决策: {compound_signals['decision']}")
    
    # 显示策略权重
    print(f"\n📈 动态权重分配:")
    for strategy, weight in compound_signals['dynamic_weights'].items():
        print(f"  {strategy}: {weight:.2%}")
    
    return engine

if __name__ == "__main__":
    test_compound_engine()