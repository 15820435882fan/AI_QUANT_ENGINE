# test_all_strategies_refactored.py
def test_all_refactored_strategies():
    """测试所有重构后的策略"""
    strategies_to_test = [
        ('SimpleMovingAverageStrategy', {
            'parameters': {'sma_fast': 10, 'sma_slow': 30}
        }),
        ('MACDStrategySmart', {
            'parameters': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
        }),
        ('BollingerBandsStrategy', {
            'parameters': {'period': 20, 'std_dev': 2.0}
        }),
        ('TurtleTradingStrategy', {
            'parameters': {'entry_period': 20, 'exit_period': 10, 'atr_period': 14}
        })
    ]
    
    for strategy_type, config in strategies_to_test:
        # 测试每个策略的创建和基本功能
        pass