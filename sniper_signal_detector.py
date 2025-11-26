# sniper_signal_detector.py
import pandas as pd
import numpy as np
import ta  # 使用ta库而不是ta-lib
from typing import Dict, Any

class SniperSignalDetector:
    """刺客信号确认系统"""
    
    def __init__(self):
        self.leverage = 10  # 10倍杠杆
        self.min_confidence = 0.65  # 降低置信度要求以便更多信号
        
    def calculate_golden_death_cross(self, df: pd.DataFrame) -> Dict[str, Any]:
        """金叉死叉信号计算"""
        if len(df) < 50:
            return {'golden_cross': False, 'death_cross': False, 'trend_strength': 0}
        
        # 使用ta库计算技术指标
        df['sma_5'] = ta.trend.sma_indicator(df['close'], window=5)
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        
        # 金叉死叉信号
        golden_cross = (df['sma_5'].iloc[-1] > df['sma_20'].iloc[-1] and 
                       df['sma_5'].iloc[-2] <= df['sma_20'].iloc[-2])
        
        death_cross = (df['sma_5'].iloc[-1] < df['sma_20'].iloc[-1] and 
                      df['sma_5'].iloc[-2] >= df['sma_20'].iloc[-2])
        
        # 趋势确认
        trend_strength = self._calculate_trend_strength(df)
        
        return {
            'golden_cross': golden_cross,
            'death_cross': death_cross,
            'trend_strength': trend_strength,
            'sma_5': df['sma_5'].iloc[-1],
            'sma_20': df['sma_20'].iloc[-1]
        }
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """计算趋势强度"""
        try:
            # MACD
            macd_line = ta.trend.macd(df['close'])
            macd_signal = 1 if macd_line.iloc[-1] > 0 else -1
            
            # RSI
            rsi = ta.momentum.rsi(df['close'], window=14)
            rsi_value = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            rsi_signal = 1 if rsi_value > 50 else -1
            
            # 布林带位置
            bb_high = ta.volatility.bollinger_hband(df['close'])
            bb_low = ta.volatility.bollinger_lband(df['close'])
            
            if bb_high.iloc[-1] - bb_low.iloc[-1] > 0:
                bb_position = (df['close'].iloc[-1] - bb_low.iloc[-1]) / (bb_high.iloc[-1] - bb_low.iloc[-1])
                bb_signal = 1 if bb_position > 0.5 else -1
            else:
                bb_signal = 0
            
            # 综合趋势强度
            trend_strength = (macd_signal + rsi_signal + bb_signal) / 3
            return trend_strength
            
        except Exception as e:
            print(f"趋势强度计算错误: {e}")
            return 0
    
    def confirm_sniper_signal(self, volume_alert: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """确认刺客交易信号"""
        try:
            # 技术指标确认
            cross_signals = self.calculate_golden_death_cross(df)
            
            # 量价确认
            volume_confirm = volume_alert['volume_ratio'] > 2.0
            price_confirm = abs(volume_alert['price_change']) > 0.015
            
            # 信号强度计算
            signal_strength = 0
            direction = 'HOLD'
            
            if cross_signals['golden_cross'] and volume_alert['price_change'] > 0:
                signal_strength = 0.7 * max(cross_signals['trend_strength'], 0.1)
                direction = 'LONG'
            elif cross_signals['death_cross'] and volume_alert['price_change'] < 0:
                signal_strength = 0.7 * max(abs(cross_signals['trend_strength']), 0.1)
                direction = 'SHORT'
            else:
                return {'confirmed': False}
            
            # 综合置信度
            confidence = (signal_strength + 
                         min(volume_alert['volume_ratio'] / 4.0, 1.0) + 
                         min(abs(volume_alert['price_change']) / 0.04, 1.0)) / 3
            
            if confidence >= self.min_confidence:
                return {
                    'confirmed': True,
                    'direction': direction,
                    'confidence': confidence,
                    'entry_price': volume_alert['current_price'],
                    'leverage': self.leverage,
                    'volume_ratio': volume_alert['volume_ratio'],
                    'price_change': volume_alert['price_change'],
                    'timestamp': pd.Timestamp.now()
                }
            
            return {'confirmed': False}
            
        except Exception as e:
            print(f"信号确认错误: {e}")
            return {'confirmed': False}

# 测试函数
def test_signal_detector():
    """测试信号检测器"""
    print("🧪 测试信号检测器...")
    
    detector = SniperSignalDetector()
    
    # 生成测试数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    prices = [100]
    for i in range(1, 100):
        change = np.random.normal(0.001, 0.01)
        prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # 测试信号
    volume_alert = {
        'volume_ratio': 3.5,
        'price_change': 0.025,
        'current_price': df['close'].iloc[-1]
    }
    
    signal = detector.confirm_sniper_signal(volume_alert, df)
    print(f"信号结果: {signal}")

if __name__ == "__main__":
    test_signal_detector()