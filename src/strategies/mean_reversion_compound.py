# src/strategies/mean_reversion_compound.py
import pandas as pd
import numpy as np
from .compound_strategy_base import CompoundStrategyBase, MarketRegime

class MeanReversionCompound(CompoundStrategyBase):
    """均值回归策略 - 复利版"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.bb_period = config.get('parameters', {}).get('bb_period', 20)
        self.bb_std = config.get('parameters', {}).get('bb_std', 2.0)
        
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算均值回归信号"""
        closes = data['close'].values
        
        if len(closes) < self.bb_period:
            return pd.DataFrame({'signal': [0] * len(data)}, index=data.index)
        
        # 布林带计算
        basis = pd.Series(closes).rolling(self.bb_period).mean()
        dev = pd.Series(closes).rolling(self.bb_period).std()
        upper = basis + self.bb_std * dev
        lower = basis - self.bb_std * dev
        
        # 生成信号
        signals = []
        for i in range(len(data)):
            if i < self.bb_period:
                signals.append(0)
            else:
                current_price = closes[i]
                # 布林带突破信号
                if current_price <= lower.iloc[i]:
                    signals.append(1)  # 下轨买入
                elif current_price >= upper.iloc[i]:
                    signals.append(-1) # 上轨卖出
                else:
                    signals.append(0)
        
        result_df = pd.DataFrame({'signal': signals}, index=data.index)
        return result_df
    
    def get_preferred_regime(self) -> list:
        """偏好震荡市场"""
        return [MarketRegime.RANGING, MarketRegime.LOW_VOL]