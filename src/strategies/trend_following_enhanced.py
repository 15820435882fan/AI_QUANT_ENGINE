# src/strategies/trend_following_enhanced.py
import pandas as pd
import numpy as np
from .compound_strategy_base import CompoundStrategyBase, MarketRegime

class TrendFollowingEnhanced(CompoundStrategyBase):
    """增强版趋势跟踪策略"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.fast_window = config.get('parameters', {}).get('fast_window', 5)   # 更敏感
        self.slow_window = config.get('parameters', {}).get('slow_window', 20)
        self.momentum_window = config.get('parameters', {}).get('momentum_window', 10)
        
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """增强信号计算"""
        closes = data['close'].values
        
        if len(closes) < self.slow_window:
            return pd.DataFrame({'signal': [0] * len(data)}, index=data.index)
        
        # 多时间框架均线
        fast_ma = pd.Series(closes).rolling(self.fast_window).mean()
        slow_ma = pd.Series(closes).rolling(self.slow_window).mean()
        
        # 动量指标
        momentum = (closes / pd.Series(closes).shift(self.momentum_window) - 1) * 100
        
        # 波动率调整
        volatility = pd.Series(closes).pct_change().rolling(20).std()
        
        signals = []
        for i in range(len(data)):
            if i < self.slow_window:
                signals.append(0)
            else:
                # 多重条件判断
                ma_signal = 1 if fast_ma.iloc[i] > slow_ma.iloc[i] else -1
                momentum_signal = 1 if momentum.iloc[i] > 0.5 else (-1 if momentum.iloc[i] < -0.5 else 0)
                
                # 综合信号（更敏感）
                if ma_signal == 1 and momentum_signal >= 0:
                    signal_strength = 0.8  # 增强信号强度
                elif ma_signal == -1 and momentum_signal <= 0:
                    signal_strength = -0.8
                elif ma_signal == 1:
                    signal_strength = 0.4  # 弱信号
                elif ma_signal == -1:
                    signal_strength = -0.4
                else:
                    signal_strength = 0
                
                # 波动率调整
                current_vol = volatility.iloc[i] if not pd.isna(volatility.iloc[i]) else 0.02
                vol_factor = 0.02 / max(current_vol, 0.01)  # 波动率越低，信号越强
                signal_strength *= min(vol_factor, 2.0)
                
                signals.append(signal_strength)
        
        result_df = pd.DataFrame({'signal': signals}, index=data.index)
        return result_df
    
    def get_preferred_regime(self) -> list:
        return [MarketRegime.TREND_UP, MarketRegime.TREND_DOWN]