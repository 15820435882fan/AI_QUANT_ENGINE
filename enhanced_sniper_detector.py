# enhanced_sniper_detector.py
import pandas as pd
import numpy as np
import ta
from typing import Dict, Any

class EnhancedSniperDetector:
    """修复版刺客信号检测器 - 修复数组形状错误"""
    
    def __init__(self):
        self.leverage = 8  # 降低杠杆
        self.min_confidence = 0.75  # 适度置信度要求
        self.volume_threshold = 3.5  # 适度成交量要求
        self.price_threshold = 0.035  # 适度价格要求
        
    def confirm_sniper_signal(self, volume_alert: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """修复版信号确认 - 修复技术分析错误"""
        try:
            # 技术指标深度分析
            technical_signals = self._advanced_technical_analysis(df)
            
            # 量价确认
            volume_confirm = volume_alert['volume_ratio'] > self.volume_threshold
            price_confirm = abs(volume_alert['price_change']) > self.price_threshold
            
            # 多重条件验证
            conditions_met = 0
            total_conditions = 6
            
            # 条件1: 成交量异常
            if volume_confirm:
                conditions_met += 1
            
            # 条件2: 价格突破
            if price_confirm:
                conditions_met += 1
            
            # 条件3: 趋势确认
            if technical_signals['trend_strength'] > 0.5:
                conditions_met += 1
            
            # 条件4: 动量确认
            if technical_signals['momentum'] > 0.02:
                conditions_met += 1
            
            # 条件5: 波动率适中
            if 0.01 < technical_signals['volatility'] < 0.05:
                conditions_met += 1
            
            # 条件6: RSI不过度超买超卖
            if 30 < technical_signals['rsi'] < 70:
                conditions_met += 1
            
            # 修复置信度计算
            base_confidence = conditions_met / total_conditions
            volume_boost = min(volume_alert['volume_ratio'] / 5.0, 0.3)
            price_boost = min(abs(volume_alert['price_change']) / 0.06, 0.2)
            
            confidence = min(base_confidence + volume_boost + price_boost, 1.0)
            
            # 确定方向
            direction = 'HOLD'
            if volume_alert['price_change'] > 0 and technical_signals['trend_strength'] > 0:
                direction = 'LONG'
            elif volume_alert['price_change'] < 0 and technical_signals['trend_strength'] < 0:
                direction = 'SHORT'
            
            if confidence >= self.min_confidence and direction != 'HOLD':
                return {
                    'confirmed': True,
                    'direction': direction,
                    'confidence': confidence,
                    'entry_price': volume_alert['current_price'],
                    'leverage': self.leverage,
                    'volume_ratio': volume_alert['volume_ratio'],
                    'price_change': volume_alert['price_change'],
                    'technical_score': technical_signals,
                    'timestamp': pd.Timestamp.now()
                }
            
            return {'confirmed': False}
            
        except Exception as e:
            print(f"信号确认错误: {e}")
            return {'confirmed': False}
    
    def _advanced_technical_analysis(self, df: pd.DataFrame) -> Dict[str, float]:
        """修复版技术分析 - 修复数组形状错误"""
        if len(df) < 50:
            return {
                'trend_strength': 0, 
                'momentum': 0, 
                'volatility': 0.02, 
                'rsi': 50
            }
        
        closes = df['close'].values
        
        # 1. 多时间框架趋势
        sma_10 = ta.trend.sma_indicator(df['close'], window=10)
        sma_30 = ta.trend.sma_indicator(df['close'], window=30)
        sma_50 = ta.trend.sma_indicator(df['close'], window=50)
        
        trend_score = 0
        if sma_10.iloc[-1] > sma_30.iloc[-1] > sma_50.iloc[-1]:
            trend_score = 0.8
        elif sma_10.iloc[-1] < sma_30.iloc[-1] < sma_50.iloc[-1]:
            trend_score = -0.8
        
        # 2. 动量指标
        momentum_5 = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
        momentum_10 = (closes[-1] - closes[-10]) / closes[-10] if len(closes) >= 10 else 0
        momentum = (momentum_5 + momentum_10) / 2
        
        # 3. 波动率 - 修复数组形状问题
        if len(closes) >= 20:
            # 使用安全的数组切片
            recent_closes = closes[-20:]
            returns = np.diff(recent_closes) / recent_closes[:-1]
            volatility = np.std(returns) if len(returns) > 0 else 0.02
        else:
            # 数据不足时使用简单计算
            if len(closes) > 1:
                returns = np.diff(closes) / closes[:-1]
                volatility = np.std(returns) if len(returns) > 0 else 0.02
            else:
                volatility = 0.02
        
        # 4. RSI
        rsi = ta.momentum.rsi(df['close'], window=14)
        rsi_value = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        # 5. MACD
        macd = ta.trend.macd_diff(df['close'])
        macd_signal = 1 if macd.iloc[-1] > 0 else -1
        
        # 综合技术分数
        technical_score = (trend_score + np.tanh(momentum * 10) + macd_signal) / 3
        
        return {
            'trend_strength': technical_score,
            'momentum': momentum,
            'volatility': volatility,
            'rsi': rsi_value
        }

# 测试函数
def test_enhanced_detector():
    """测试修复版信号检测器"""
    print("🧪 测试修复版信号检测器...")
    
    detector = EnhancedSniperDetector()
    
    # 测试数据
    test_alert = {
        'volume_ratio': 4.2,
        'price_change': 0.045,
        'current_price': 50000.0
    }
    
    # 生成测试DataFrame
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5T')
    prices = 50000 + np.random.normal(0, 1000, 100).cumsum()
    
    test_df = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.normal(50, 10, 100),
        'low': prices - np.random.normal(50, 10, 100),
        'close': prices,
        'volume': np.random.randint(10000, 50000, 100)
    }, index=dates)
    
    signal = detector.confirm_sniper_signal(test_alert, test_df)
    
    print(f"📊 信号检测结果:")
    print(f"  确认: {signal['confirmed']}")
    if signal['confirmed']:
        print(f"  方向: {signal['direction']}")
        print(f"  置信度: {signal['confidence']:.1%}")
        print(f"  入场价格: ${signal['entry_price']:.2f}")
        print(f"  杠杆: {signal['leverage']}x")
        print(f"  成交量比率: {signal['volume_ratio']:.2f}")
        print(f"  价格变动: {signal['price_change']:.2%}")
    
    return detector

if __name__ == "__main__":
    test_enhanced_detector()