# test_enhanced_engine.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enhanced_compound_engine import EnhancedCompoundEngine
from src.strategies.trend_following_enhanced import TrendFollowingEnhanced
from src.strategies.mean_reversion_compound import MeanReversionCompound

def generate_test_data(periods: int = 100) -> pd.DataFrame:
    """生成测试数据"""
    np.random.seed(42)
    
    prices = [100]
    trend = 0.001
    volatility = 0.02
    
    for i in range(1, periods):
        change = np.random.normal(trend, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 10))
    
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

def test_enhanced_engine():
    """测试增强版引擎"""
    print("🚀 测试增强版复利引擎...")
    
    engine = EnhancedCompoundEngine(initial_capital=10000.0)
    
    # 使用增强版策略
    trend_strategy = TrendFollowingEnhanced({
        'name': '增强趋势',
        'weight': 0.5,
        'parameters': {'fast_window': 5, 'slow_window': 15, 'momentum_window': 10}
    })
    
    mean_reversion_strategy = MeanReversionCompound({
        'name': '均值回归', 
        'weight': 0.5,
        'parameters': {'bb_period': 15, 'bb_std': 1.8}  # 更敏感的参数
    })
    
    engine.add_strategy(trend_strategy)
    engine.add_strategy(mean_reversion_strategy)
    
    # 测试数据
    test_data = generate_test_data(100)
    
    # 测试信号生成
    signals = engine.generate_compound_signals(test_data)
    
    print(f"\n🎯 增强版信号结果:")
    print(f"  最终信号: {signals['final_signal']:.3f}")
    print(f"  综合置信度: {signals['combined_confidence']:.2f}")
    print(f"  市场状态: {signals['market_regime']}")
    print(f"  动态阈值: {signals['decision']['dynamic_thresholds']}")
    print(f"  交易决策: {signals['decision']['action']}")
    print(f"  仓位大小: {signals['decision']['position_size']:.1%}")
    print(f"  决策原因: {signals['decision']['reason']}")
    
    # 显示策略权重
    print(f"\n📈 动态权重分配:")
    for strategy, weight in signals['dynamic_weights'].items():
        print(f"  {strategy}: {weight:.1%}")
    
    return engine

if __name__ == "__main__":
    test_enhanced_engine()