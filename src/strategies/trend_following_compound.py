# src/strategies/trend_following_compound.py
import pandas as pd
import numpy as np
from .compound_strategy_base import CompoundStrategyBase, MarketRegime

class TrendFollowingCompound(CompoundStrategyBase):
    """趋势跟踪策略 - 复利版"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.fast_window = config.get('parameters', {}).get('fast_window', 10)
        self.slow_window = config.get('parameters', {}).get('slow_window', 30)
        
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算趋势信号"""
        closes = data['close'].values
        
        if len(closes) < self.slow_window:
            return pd.DataFrame({'signal': [0] * len(data)}, index=data.index)
        
        # 计算双均线
        fast_ma = pd.Series(closes).rolling(self.fast_window).mean()
        slow_ma = pd.Series(closes).rolling(self.slow_window).mean()
        
        # 生成信号
        signals = []
        for i in range(len(data)):
            if i < self.slow_window:
                signals.append(0)
            else:
                # 均线交叉信号
                if fast_ma.iloc[i] > slow_ma.iloc[i] and fast_ma.iloc[i-1] <= slow_ma.iloc[i-1]:
                    signals.append(1)  # 金叉买入
                elif fast_ma.iloc[i] < slow_ma.iloc[i] and fast_ma.iloc[i-1] >= slow_ma.iloc[i-1]:
                    signals.append(-1) # 死叉卖出
                else:
                    signals.append(signals[-1] if i > 0 else 0)
        
        result_df = pd.DataFrame({'signal': signals}, index=data.index)
        return result_df
    
    def get_preferred_regime(self) -> list:
        """偏好趋势市场"""
        return [MarketRegime.TREND_UP, MarketRegime.TREND_DOWN]