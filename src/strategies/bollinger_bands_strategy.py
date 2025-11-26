# src/strategies/bollinger_bands_strategy.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .strategy_orchestrator import BaseStrategy

class BollingerBandsStrategy(BaseStrategy):
    """布林带策略 - 统一构造函数版本"""
    
    def __init__(self, config: dict, data_provider=None):
        super().__init__(config, data_provider)
        
        self.period = self.parameters.get('period', 20)
        self.std_dev = self.parameters.get('std_dev', 2.0)
        
    @staticmethod
    def get_required_parameters() -> List[str]:
        return ['period', 'std_dev']
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算布林带信号"""
        if len(data) < self.period:
            return pd.DataFrame()
            
        data = data.copy()
        
        # 布林带计算
        data['sma'] = data['close'].rolling(window=self.period).mean()
        data['std'] = data['close'].rolling(window=self.period).std()
        data['upper_band'] = data['sma'] + (data['std'] * self.std_dev)
        data['lower_band'] = data['sma'] - (data['std'] * self.std_dev)
        
        # 生成信号
        data['signal'] = 0
        data.loc[data['close'] < data['lower_band'], 'signal'] = 1    # 超卖买入
        data.loc[data['close'] > data['upper_band'], 'signal'] = -1   # 超买卖出
        
        return data