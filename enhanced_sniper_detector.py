import pandas as pd
import numpy as np
import talib as ta
from typing import Dict, List, Optional
import logging

class EnhancedSniperDetector:
    """
    增强版信号检测器 - 优化版本
    多重技术指标确认系统，目标胜率35%+
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 优化后的默认参数
        self.default_params = {
            'volume_threshold': 2.0,
            'price_breakout_period': 12,
            'rsi_oversold': 40,
            'rsi_overbought': 70,
            'min_signal_strength': 4,
            'ema_fast_period': 12,
            'ema_slow_period': 26,
            'rsi_period': 14,
            'bb_period': 20
        }
        
    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """
        生成交易信号 - 优化版
        使用多重技术指标确认系统
        """
        if data.empty or len(data) < 50:
            return {'signal': 'HOLD', 'strength': 0, 'reason': '数据不足'}
        
        try:
            # 1. 计算市场波动率用于动态参数
            market_volatility = self._calculate_market_volatility(data)
            dynamic_params = self.get_dynamic_parameters(market_volatility)
            
            # 2. 多重技术指标确认
            signal_strength = 0
            signal_reasons = []
            
            # 基础量价突破
            volume_breakout = self._check_volume_breakout(data, dynamic_params)
            if volume_breakout:
                signal_strength += 1
                signal_reasons.append("成交量突破")
            
            price_breakout = self._check_price_breakout(data, dynamic_params)
            if price_breakout:
                signal_strength += 1
                signal_reasons.append("价格突破")
            
            # 新增技术指标确认
            trend_confirmation = self._check_trend_direction(data)
            if trend_confirmation:
                signal_strength += 1
                signal_reasons.append("趋势确认")
            
            momentum_confirmation = self._check_momentum_strength(data)
            if momentum_confirmation:
                signal_strength += 1
                signal_reasons.append("动量确认")
            
            volatility_filter = self._check_volatility_condition(data)
            if volatility_filter:
                signal_strength += 1
                signal_reasons.append("波动率过滤通过")
            
            # 3. 信号决策
            min_strength = dynamic_params['min_signal_strength']
            
            if signal_strength >= min_strength:
                # 额外确认：价格必须在关键均线上方
                price_above_ema = self._check_price_above_ema(data)
                if price_above_ema:
                    return {
                        'signal': 'BUY',
                        'strength': signal_strength,
                        'reasons': signal_reasons,
                        'volatility': market_volatility,
                        'timestamp': data.index[-1]
                    }
            
            # 卖出信号逻辑（可根据需要扩展）
            sell_signal = self._generate_sell_signal(data, signal_strength)
            if sell_signal:
                return {
                    'signal': 'SELL', 
                    'strength': signal_strength,
                    'reasons': ['卖出条件触发'],
                    'timestamp': data.index[-1]
                }
            
            return {
                'signal': 'HOLD',
                'strength': signal_strength,
                'reasons': signal_reasons,
                'timestamp': data.index[-1]
            }
            
        except Exception as e:
            self.logger.error(f"信号生成错误: {e}")
            return {'signal': 'HOLD', 'strength': 0, 'reason': f'错误: {str(e)}'}
    
    def _calculate_market_volatility(self, data: pd.DataFrame) -> float:
        """
        计算市场波动率 - 使用ATR标准化
        """
        try:
            atr = ta.ATR(data['high'], data['low'], data['close'], timeperiod=14)
            if len(atr) > 0 and not np.isnan(atr.iloc[-1]):
                current_atr = atr.iloc[-1]
                current_price = data['close'].iloc[-1]
                return current_atr / current_price
            return 0.03  # 默认波动率
        except:
            return 0.03
    
    def get_dynamic_parameters(self, market_volatility: float) -> Dict:
        """
        根据市场波动率动态调整参数
        """
        if market_volatility > 0.05:  # 高波动市场
            return {
                'volume_threshold': 2.2,
                'price_breakout_period': 10,
                'min_signal_strength': 4,  # 需要更强的确认
                'rsi_oversold': 35,
                'rsi_overbought': 75
            }
        elif market_volatility < 0.02:  # 低波动市场
            return {
                'volume_threshold': 1.8,
                'price_breakout_period': 15,
                'min_signal_strength': 3,  # 可以适当放宽
                'rsi_oversold': 45,
                'rsi_overbought': 65
            }
        else:  # 正常波动
            return {
                'volume_threshold': 2.0,
                'price_breakout_period': 12,
                'min_signal_strength': 4,
                'rsi_oversold': 40,
                'rsi_overbought': 70
            }
    
    def _check_volume_breakout(self, data: pd.DataFrame, params: Dict) -> bool:
        """
        成交量突破检查 - 优化版
        """
        try:
            if 'volume' not in data.columns:
                return False
                
            current_volume = data['volume'].iloc[-1]
            volume_ma = data['volume'].rolling(window=20).mean()
            
            if len(volume_ma) < 20 or np.isnan(volume_ma.iloc[-1]):
                return False
            
            volume_ratio = current_volume / volume_ma.iloc[-1]
            return volume_ratio > params['volume_threshold']
            
        except Exception as e:
            self.logger.warning(f"成交量突破检查错误: {e}")
            return False
    
    def _check_price_breakout(self, data: pd.DataFrame, params: Dict) -> bool:
        """
        价格突破检查 - 优化版
        """
        try:
            period = params['price_breakout_period']
            if len(data) < period + 1:
                return False
            
            current_high = data['high'].iloc[-1]
            previous_highs = data['high'].iloc[-(period+1):-1]
            
            # 突破近期高点
            breakout_condition = current_high > previous_highs.max()
            
            # 确认突破有效性：收盘价也突破
            current_close = data['close'].iloc[-1]
            close_breakout = current_close > previous_highs.max() * 0.998
            
            return breakout_condition and close_breakout
            
        except Exception as e:
            self.logger.warning(f"价格突破检查错误: {e}")
            return False
    
    def _check_trend_direction(self, data: pd.DataFrame) -> bool:
        """
        趋势方向确认 - 使用EMA双线确认趋势
        """
        try:
            ema_fast = ta.EMA(data['close'], timeperiod=self.default_params['ema_fast_period'])
            ema_slow = ta.EMA(data['close'], timeperiod=self.default_params['ema_slow_period'])
            
            if len(ema_fast) < 2 or len(ema_slow) < 2:
                return False
            
            # 当前趋势
            current_fast = ema_fast.iloc[-1]
            current_slow = ema_slow.iloc[-1]
            current_trend = current_fast > current_slow
            
            # 前期趋势
            prev_fast = ema_fast.iloc[-2]
            prev_slow = ema_slow.iloc[-2] 
            prev_trend = prev_fast > prev_slow
            
            # 趋势确认：当前向上且与前一期一致
            trend_consistent = current_trend == prev_trend
            
            # 趋势强度：快线与慢线的距离
            trend_strength = (current_fast - current_slow) / current_slow > 0.005
            
            return current_trend and trend_consistent and trend_strength
            
        except Exception as e:
            self.logger.warning(f"趋势方向检查错误: {e}")
            return False
    
    def _check_momentum_strength(self, data: pd.DataFrame) -> bool:
        """
        动量强度确认 - 使用RSI和MACD双重确认
        """
        try:
            # RSI动量过滤
            rsi = ta.RSI(data['close'], timeperiod=self.default_params['rsi_period'])
            if len(rsi) == 0 or np.isnan(rsi.iloc[-1]):
                return False
            
            current_rsi = rsi.iloc[-1]
            rsi_condition = (self.default_params['rsi_oversold'] < current_rsi < 
                           self.default_params['rsi_overbought'])
            
            # MACD动量确认
            macd, macd_signal, _ = ta.MACD(data['close'])
            macd_condition = False
            if (len(macd) > 1 and len(macd_signal) > 1 and 
                not np.isnan(macd.iloc[-1]) and not np.isnan(macd_signal.iloc[-1])):
                
                current_macd = macd.iloc[-1]
                current_signal = macd_signal.iloc[-1]
                prev_macd = macd.iloc[-2]
                
                # MACD在信号线上方且为正值，且动量增强
                macd_above_signal = current_macd > current_signal
                macd_positive = current_macd > 0
                momentum_increasing = current_macd > prev_macd
                
                macd_condition = macd_above_signal and macd_positive and momentum_increasing
            
            return rsi_condition and macd_condition
            
        except Exception as e:
            self.logger.warning(f"动量强度检查错误: {e}")
            return False
    
    def _check_volatility_condition(self, data: pd.DataFrame) -> bool:
        """
        波动率过滤 - 使用布林带识别合适的波动环境
        """
        try:
            bb_upper, bb_middle, bb_lower = ta.BBANDS(
                data['close'], 
                timeperiod=self.default_params['bb_period']
            )
            
            if (len(bb_upper) == 0 or len(bb_lower) == 0 or 
                np.isnan(bb_upper.iloc[-1]) or np.isnan(bb_lower.iloc[-1])):
                return False
            
            current_close = data['close'].iloc[-1]
            current_upper = bb_upper.iloc[-1]
            current_lower = bb_lower.iloc[-1]
            current_middle = bb_middle.iloc[-1]
            
            # 布林带宽度过滤（波动率）
            bb_width = (current_upper - current_lower) / current_middle
            volatility_ok = 0.02 < bb_width < 0.08
            
            # 价格位置过滤：不在布林带极端位置
            position_ok = (current_close > current_lower * 1.02 and 
                          current_close < current_upper * 0.98)
            
            # 带宽趋势：布林带不能正在收缩（避免即将突破）
            if len(bb_upper) > 5:
                recent_widths = [(bb_upper.iloc[-i] - bb_lower.iloc[-i]) / bb_middle.iloc[-i] 
                               for i in range(1, 6)]
                width_decreasing = all(recent_widths[i] >= recent_widths[i+1] 
                                     for i in range(len(recent_widths)-1))
                bandwidth_ok = not width_decreasing
            else:
                bandwidth_ok = True
            
            return volatility_ok and position_ok and bandwidth_ok
            
        except Exception as e:
            self.logger.warning(f"波动率条件检查错误: {e}")
            return False
    
    def _check_price_above_ema(self, data: pd.DataFrame) -> bool:
        """
        价格在关键EMA上方确认
        """
        try:
            ema20 = ta.EMA(data['close'], timeperiod=20)
            if len(ema20) > 0 and not np.isnan(ema20.iloc[-1]):
                return data['close'].iloc[-1] > ema20.iloc[-1]
            return False
        except:
            return False
    
    def _generate_sell_signal(self, data: pd.DataFrame, buy_strength: int) -> bool:
        """
        生成卖出信号 - 基础版本（可根据需要扩展）
        """
        # 这里可以添加复杂的卖出逻辑
        # 目前使用简单规则：当买入信号强度很低时考虑卖出
        return buy_strength <= 1
    
    def batch_detect(self, data_list: List[pd.DataFrame]) -> List[Dict]:
        """
        批量检测信号
        """
        results = []
        for data in data_list:
            signal = self.generate_signals(data)
            results.append(signal)
        return results

# 使用示例
if __name__ == "__main__":
    # 测试代码
    detector = EnhancedSniperDetector()
    
    # 创建测试数据
    sample_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
        'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'volume': [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]
    })
    
    signal = detector.generate_signals(sample_data)
    print("测试信号:", signal)