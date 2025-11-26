# src/strategies/turtle_trading_strategy.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .strategy_orchestrator import BaseStrategy

class TurtleTradingStrategy(BaseStrategy):
    """海龟交易策略 - 统一构造函数版本"""
    
    def __init__(self, config: dict, data_provider=None):
        super().__init__(config, data_provider)
        
        self.entry_period = self.parameters.get('entry_period', 20)
        self.exit_period = self.parameters.get('exit_period', 10)
        self.atr_period = self.parameters.get('atr_period', 14)
        
    @staticmethod
    def get_required_parameters() -> List[str]:
        return ['entry_period', 'exit_period', 'atr_period']
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算海龟交易信号"""
        if len(data) < max(self.entry_period, self.exit_period, self.atr_period):
            return pd.DataFrame()
            
        data = data.copy()
        
        # 海龟策略计算
        data['high_entry'] = data['high'].rolling(window=self.entry_period).max()
        data['low_exit'] = data['low'].rolling(window=self.exit_period).min()
        
        # ATR计算
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = np.maximum(np.maximum(high_low, high_close), low_close)
        data['atr'] = true_range.rolling(window=self.atr_period).mean()
        
        # 生成信号
        data['signal'] = 0
        data.loc[data['close'] > data['high_entry'], 'signal'] = 1    # 突破入场
        data.loc[data['close'] < data['low_exit'], 'signal'] = -1     # 突破出场
        
        return data