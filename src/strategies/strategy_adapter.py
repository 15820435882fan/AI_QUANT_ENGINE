# 新建：src/strategies/strategy_adapter.py

class StrategyAdapter:
    """策略适配器，处理新旧接口的兼容性问题"""
    
    @staticmethod
    def convert_legacy_strategy(legacy_strategy, new_config: dict):
        """
        将旧策略实例转换为新接口兼容的实例
        """
        strategy_type = legacy_strategy.__class__.__name__
        
        # 构建适配的config
        adapted_config = {
            'name': new_config.get('name', strategy_type),
            'symbols': new_config.get('symbols', ['BTC/USDT']),
            'parameters': StrategyAdapter._extract_parameters(legacy_strategy)
        }
        
        return adapted_config
    
    @staticmethod
    def _extract_parameters(strategy_instance):
        """从旧策略实例中提取参数"""
        params = {}
        
        # 基于策略类型提取特定参数
        strategy_class = strategy_instance.__class__.__name__
        
        if strategy_class == 'SimpleMovingAverageStrategy':
            params = {
                'sma_fast': getattr(strategy_instance, 'fast_period', 10),
                'sma_slow': getattr(strategy_instance, 'slow_period', 20)
            }
        elif strategy_class == 'MACDStrategySmart':
            params = {
                'fast_period': getattr(strategy_instance, 'fast_period', 12),
                'slow_period': getattr(strategy_instance, 'slow_period', 26),
                'signal_period': getattr(strategy_instance, 'signal_period', 9)
            }
        # ... 其他策略类型的参数提取
        
        return params