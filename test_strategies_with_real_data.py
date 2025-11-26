# test_strategies_with_real_data.py
import pandas as pd
import numpy as np
from src.strategies.strategy_factory import strategy_factory

def generate_realistic_test_data(length=100):
    """生成更真实的测试数据"""
    np.random.seed(42)
    
    # 生成价格数据
    prices = [100.0]
    for i in range(1, length):
        change = np.random.normal(0, 2)  # 随机波动
        new_price = prices[-1] + change
        prices.append(max(new_price, 1))  # 确保价格为正
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p + abs(np.random.normal(0, 1)) for p in prices],
        'low': [p - abs(np.random.normal(0, 1)) for p in prices], 
        'close': prices,
        'volume': [np.random.randint(1000, 10000) for _ in prices]
    })
    
    return data

def test_all_strategies_with_data():
    """使用真实数据测试所有策略"""
    print("🧪 使用真实数据测试所有策略...")
    
    # 生成足够长度的测试数据
    test_data = generate_realistic_test_data(100)
    print(f"📊 测试数据形状: {test_data.shape}")
    
    strategies_config = [
        ('SimpleMovingAverageStrategy', {
            'name': 'SMA测试',
            'parameters': {'sma_fast': 10, 'sma_slow': 30}
        }),
        ('MACDStrategySmart', {
            'name': 'MACD测试', 
            'parameters': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
        }),
        ('BollingerBandsStrategy', {
            'name': '布林带测试',
            'parameters': {'period': 20, 'std_dev': 2.0}
        }),
        ('TurtleTradingStrategy', {
            'name': '海龟测试',
            'parameters': {'entry_period': 20, 'exit_period': 10, 'atr_period': 14}
        })
    ]
    
    results = {}
    
    for strategy_type, config in strategies_config:
        try:
            print(f"\n🔍 测试 {strategy_type}...")
            
            # 创建策略
            strategy = strategy_factory.create_strategy(strategy_type, config)
            
            # 计算信号
            signals = strategy.calculate_signals(test_data)
            
            if not signals.empty:
                latest_signal = signals['signal'].iloc[-1]
                signal_count = (signals['signal'] != 0).sum()
                results[strategy_type] = {
                    'status': '✅ 成功',
                    'latest_signal': latest_signal,
                    'signal_count': signal_count,
                    'data_shape': signals.shape
                }
                print(f"  ✅ 信号生成成功 - 最新信号: {latest_signal}, 信号数量: {signal_count}")
            else:
                results[strategy_type] = {'status': '⚠️ 无信号'}
                print(f"  ⚠️ 无信号生成")
                
        except Exception as e:
            results[strategy_type] = {'status': f'❌ 失败: {e}'}
            print(f"  ❌ 测试失败: {e}")
    
    # 输出总结
    print(f"\n📊 测试总结:")
    for strategy, result in results.items():
        print(f"  {strategy}: {result['status']}")

if __name__ == "__main__":
    test_all_strategies_with_data()