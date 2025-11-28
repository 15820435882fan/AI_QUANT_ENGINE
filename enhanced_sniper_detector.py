# enhanced_sniper_detector.py - 添加缺失的方法
import pandas as pd
import numpy as np
from datetime import datetime

class EnhancedSniperDetector:
    def __init__(self):
        self.logger = logging.getLogger('EnhancedSniperDetector')
    
    def analyze_enhanced_signals(self, data, symbol):
        """分析增强信号 - 修复版本"""
        try:
            if data is None or len(data) < 20:
                return pd.DataFrame()
            
            # 复制数据避免修改原始数据
            df = data.copy()
            
            # 确保数据格式正确
            if 'close' not in df.columns:
                self.logger.error("数据缺少close列")
                return pd.DataFrame()
            
            # 计算技术指标
            df = self._calculate_technical_indicators(df)
            
            # 生成信号
            signals = self._generate_trading_signals(df)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"信号分析错误: {e}")
            return pd.DataFrame()
    
    def _calculate_technical_indicators(self, df):
        """计算技术指标"""
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
        
        # 布林带
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        
        # 移动平均线
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # 价格动量
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change() if 'volume' in df.columns else 0
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """计算MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """计算布林带"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def _generate_trading_signals(self, df):
        """生成交易信号"""
        signals = []
        
        for i in range(len(df)):
            if i < 50:  # 确保有足够数据
                signals.append({'signal_strength': 0, 'signal_type': 'HOLD'})
                continue
                
            row = df.iloc[i]
            signal_strength = 0
            
            # RSI信号
            if row['rsi'] < 30:  # 超卖
                signal_strength += 0.3
            elif row['rsi'] > 70:  # 超买
                signal_strength -= 0.3
            
            # MACD信号
            if row['macd'] > row['macd_signal'] and row['macd_hist'] > 0:
                signal_strength += 0.2
            elif row['macd'] < row['macd_signal'] and row['macd_hist'] < 0:
                signal_strength -= 0.2
            
            # 布林带信号
            if row['close'] < row['bb_lower']:  # 价格触及下轨
                signal_strength += 0.2
            elif row['close'] > row['bb_upper']:  # 价格触及上轨
                signal_strength -= 0.2
            
            # 移动平均线信号
            if row['sma_20'] > row['sma_50']:  # 短期均线上穿长期
                signal_strength += 0.2
            else:
                signal_strength -= 0.1
            
            # 确定信号类型
            if signal_strength > 0.5:
                signal_type = 'STRONG_BUY'
            elif signal_strength > 0.2:
                signal_type = 'BUY'
            elif signal_strength < -0.5:
                signal_type = 'STRONG_SELL'
            elif signal_strength < -0.2:
                signal_type = 'SELL'
            else:
                signal_type = 'HOLD'
            
            signals.append({
                'signal_strength': signal_strength,
                'signal_type': signal_type,
                'timestamp': row.name if hasattr(row, 'name') else datetime.now()
            })
        
        return pd.DataFrame(signals)