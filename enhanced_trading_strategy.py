# enhanced_trading_strategy.py
class EnhancedTradingStrategy:
    def __init__(self, base_position=0.1, dynamic_sizing=True):
        self.base_position = base_position
        self.dynamic_sizing = dynamic_sizing
        self.performance_tracker = {
            'recent_trades': [],
            'win_streak': 0,
            'loss_streak': 0
        }
    
    def calculate_enhanced_signals(self, df):
        """增强信号生成"""
        signals = []
        
        for i in range(50, len(df)):
            current = df.iloc[i]
            
            # 基础信号 (从可信回测移植)
            signal_strength = 0
            
            # RSI增强
            if current['rsi'] < 25:  # 更严格的超卖
                signal_strength += 0.4
            elif current['rsi'] > 75:  # 更严格的超买
                signal_strength -= 0.4
            elif current['rsi'] < 35:
                signal_strength += 0.2
            elif current['rsi'] > 65:
                signal_strength -= 0.2
            
            # MACD增强
            macd_trend = current['macd'] - current['macd_signal']
            if macd_trend > 0:
                signal_strength += 0.2
            else:
                signal_strength -= 0.2
            
            # 趋势确认
            price_vs_ma = current['close'] / current['sma_20'] - 1
            if abs(price_vs_ma) > 0.02:  # 价格偏离均线2%
                if price_vs_ma > 0:
                    signal_strength -= 0.1  # 可能回调
                else:
                    signal_strength += 0.1  # 可能反弹
            
            # 生成信号
            if abs(signal_strength) > 0.3:
                signals.append({
                    'timestamp': current.name,
                    'price': current['close'],
                    'signal': 'BUY' if signal_strength > 0 else 'SELL',
                    'strength': min(abs(signal_strength), 0.8),
                    'rsi': current['rsi'],
                    'macd_trend': macd_trend
                })
        
        return signals
    
    def dynamic_position_size(self, signal_strength, recent_performance):
        """动态仓位调整"""
        base_size = self.base_position
        
        # 根据信号强度调整
        strength_multiplier = 0.5 + signal_strength  # 0.5-1.3倍
        
        # 根据近期表现调整
        if len(recent_performance) >= 5:
            recent_win_rate = len([p for p in recent_performance if p > 0]) / len(recent_performance)
            if recent_win_rate > 0.7:
                performance_multiplier = 1.2
            elif recent_win_rate < 0.4:
                performance_multiplier = 0.7
            else:
                performance_multiplier = 1.0
        else:
            performance_multiplier = 1.0
        
        final_size = base_size * strength_multiplier * performance_multiplier
        return min(final_size, 0.25)  # 最大25%仓位