import pandas as pd
import numpy as np
import talib as ta

class SimpleSniperDetector:
    """
    简化版高性能信号检测器
    基于原始成功版本优化
    """
    
    def __init__(self):
        self.volume_threshold = 1.8
        self.price_period = 10
        
    def generate_signals(self, data):
        """简化但有效的信号生成"""
        if len(data) < 20:
            return {'signal': 'HOLD', 'strength': 0}
        
        try:
            # 1. 成交量突破（核心）
            volume_signal = self._volume_breakout(data)
            
            # 2. 价格突破（核心）  
            price_signal = self._price_breakout(data)
            
            # 3. 简单趋势确认
            trend_signal = self._simple_trend(data)
            
            # 信号强度计算
            strength = sum([volume_signal, price_signal, trend_signal])
            
            if strength >= 2:  # 2/3条件满足即可
                return {
                    'signal': 'BUY',
                    'strength': strength,
                    'reasons': ['简化高效信号']
                }
            else:
                return {
                    'signal': 'SELL' if strength == 0 else 'HOLD',
                    'strength': strength
                }
                
        except Exception as e:
            return {'signal': 'HOLD', 'strength': 0, 'error': str(e)}
    
    def _volume_breakout(self, data):
        """成交量突破 - 简化版"""
        if 'volume' not in data.columns:
            return 0
            
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].tail(20).mean()
        
        return 1 if current_volume > avg_volume * self.volume_threshold else 0
    
    def _price_breakout(self, data):
        """价格突破 - 简化版"""
        if len(data) < self.price_period + 1:
            return 0
            
        current_high = data['high'].iloc[-1]
        previous_highs = data['high'].iloc[-(self.price_period+1):-1]
        
        return 1 if current_high > previous_highs.max() else 0
    
    def _simple_trend(self, data):
        """简单趋势确认"""
        if len(data) < 20:
            return 0
            
        ema_fast = ta.EMA(data['close'], timeperiod=8)
        ema_slow = ta.EMA(data['close'], timeperiod=21)
        
        if len(ema_fast) == 0 or len(ema_slow) == 0:
            return 0
            
        return 1 if ema_fast.iloc[-1] > ema_slow.iloc[-1] else 0