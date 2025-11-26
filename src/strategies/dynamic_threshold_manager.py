# src/strategies/dynamic_threshold_manager.py
class DynamicThresholdManager:
    """动态阈值管理器"""
    
    def __init__(self):
        self.market_volatility = 0.02
        self.base_buy_threshold = 0.1
        self.base_sell_threshold = -0.1
        
    def calculate_dynamic_thresholds(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算动态交易阈值"""
        closes = data['close'].values
        
        # 计算当前波动率
        returns = np.diff(closes) / closes[:-1]
        current_volatility = np.std(returns) if len(returns) > 0 else 0.02
        
        # 波动率调整因子
        vol_factor = current_volatility / self.market_volatility
        
        # 动态调整阈值
        buy_threshold = self.base_buy_threshold * vol_factor
        sell_threshold = self.base_sell_threshold * vol_factor
        
        # 确保阈值在合理范围内
        buy_threshold = max(0.05, min(buy_threshold, 0.3))
        sell_threshold = min(-0.05, max(sell_threshold, -0.3))
        
        return {
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'volatility_factor': vol_factor
        }