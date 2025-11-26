# src/strategies/macd_strategy_smart.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .strategy_orchestrator import BaseStrategy

class MACDStrategySmart(BaseStrategy):
    """MACD智能策略 - 统一构造函数版本"""
    
    def __init__(self, config: dict, data_provider=None):
        super().__init__(config, data_provider)
        
        # 从parameters获取参数
        self.fast_period = self.parameters.get('fast_period', 12)
        self.slow_period = self.parameters.get('slow_period', 26) 
        self.signal_period = self.parameters.get('signal_period', 9)
        
    def _handle_backward_compatibility(self):
        """MACD特定向后兼容性处理"""
        # 映射旧参数名
        param_mapping = {
            'fast_period': 'fast_period',
            'slow_period': 'slow_period', 
            'signal_period': 'signal_period'
        }
        
        for old_param, new_param in param_mapping.items():
            if old_param in self.config and new_param not in self.parameters:
                self.parameters[new_param] = self.config[old_param]

    @staticmethod
    def get_required_parameters() -> List[str]:
        return ['fast_period', 'slow_period', 'signal_period']
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算MACD信号"""
        if len(data) < self.slow_period:
            return pd.DataFrame()
            
        data = data.copy()
        
        # MACD计算
        exp1 = data['close'].ewm(span=self.fast_period).mean()
        exp2 = data['close'].ewm(span=self.slow_period).mean()
        data['macd'] = exp1 - exp2
        data['signal_line'] = data['macd'].ewm(span=self.signal_period).mean()
        data['histogram'] = data['macd'] - data['signal_line']
        
        # 生成信号
        data['signal'] = 0
        data.loc[data['macd'] > data['signal_line'], 'signal'] = 1
        data.loc[data['macd'] < data['signal_line'], 'signal'] = -1
        
        return data