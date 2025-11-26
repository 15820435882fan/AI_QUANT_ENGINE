# src/strategies/market_regime_detector.py
import pandas as pd
import numpy as np
from typing import Dict, List
from .compound_strategy_base import MarketRegime

class MarketRegimeDetector:
    """市场状态检测器 - 核心组件"""
    
    def __init__(self, lookback_period: int = 50):
        self.lookback_period = lookback_period
        
    def detect_regime(self, data: pd.DataFrame) -> Dict[str, any]:
        """检测当前市场状态"""
        if len(data) < self.lookback_period:
            return {'regime': MarketRegime.RANGING, 'confidence': 0.5}
            
        closes = data['close'].values
        volumes = data['volume'].values
        
        # 计算技术指标
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.02
        
        # 趋势判断
        sma_short = np.mean(closes[-10:])
        sma_long = np.mean(closes[-30:])
        trend_strength = (sma_short - sma_long) / sma_long
        
        # 动量判断
        momentum = (closes[-1] - closes[-10]) / closes[-10] if len(closes) >= 10 else 0
        
        # 波动率判断
        volume_change = np.std(volumes[-10:]) / np.mean(volumes[-10:]) if len(volumes) >= 10 else 0
        
        # 状态判断逻辑
        regime = MarketRegime.RANGING
        confidence = 0.5
        
        if abs(trend_strength) > 0.02:  # 2%以上的趋势
            if trend_strength > 0:
                regime = MarketRegime.TREND_UP
                confidence = min(abs(trend_strength) * 10, 0.9)
            else:
                regime = MarketRegime.TREND_DOWN  
                confidence = min(abs(trend_strength) * 10, 0.9)
        elif volatility > 0.03:  # 高波动
            regime = MarketRegime.HIGH_VOL
            confidence = min(volatility * 10, 0.8)
        elif volatility < 0.01:  # 低波动
            regime = MarketRegime.LOW_VOL
            confidence = 0.7
            
        return {
            'regime': regime,
            'confidence': confidence,
            'trend_strength': trend_strength,
            'volatility': volatility,
            'momentum': momentum
        }