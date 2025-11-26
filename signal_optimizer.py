# signal_optimizer.py
import pandas as pd
import numpy as np
from production_trading_system import ProductionTradingSystem
from test_strategies_with_real_data import generate_realistic_test_data

class SignalOptimizer:
    """信号阈值优化器"""
    
    def __init__(self):
        self.trading_system = ProductionTradingSystem()
        self.optimization_results = []
    
    def optimize_signal_thresholds(self, historical_data: pd.DataFrame):
        """优化信号触发阈值"""
        print("优化信号阈值...")
        
        # 测试不同的阈值组合
        threshold_combinations = [
            {'buy_threshold': 0.2, 'sell_threshold': -0.2},
            {'buy_threshold': 0.3, 'sell_threshold': -0.3},
            {'buy_threshold': 0.4, 'sell_threshold': -0.4},
            {'buy_threshold': 0.1, 'sell_threshold': -0.1},
        ]
        
        best_score = -np.inf
        best_thresholds = None
        
        for thresholds in threshold_combinations:
            score = self._evaluate_thresholds(thresholds, historical_data)
            self.optimization_results.append({
                'thresholds': thresholds,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_thresholds = thresholds
            
            print(f"  阈值 {thresholds}: 得分 {score:.4f}")
        
        print(f"🎯 最佳阈值: {best_thresholds}, 得分: {best_score:.4f}")
        return best_thresholds
    
    def _evaluate_thresholds(self, thresholds: dict, data: pd.DataFrame) -> float:
        """评估阈值性能"""
        # 模拟交易决策
        buy_threshold = thresholds['buy_threshold']
        sell_threshold = thresholds['sell_threshold']
        
        # 使用优化后的策略
        self.trading_system.initialize_optimized_strategies(data)
        signals = self.trading_system.strategy_manager.calculate_combined_signals(data)
        
        if signals.empty:
            return -np.inf
        
        combined_signal = signals['combined_signal']
        
        # 计算交易信号质量
        buy_signals = (combined_signal > buy_threshold).sum()
        sell_signals = (combined_signal < sell_threshold).sum()
        hold_signals = ((combined_signal >= sell_threshold) & (combined_signal <= buy_threshold)).sum()
        
        # 评分标准：适当的交易频率 + 信号清晰度
        total_periods = len(combined_signal)
        trade_frequency = (buy_signals + sell_signals) / total_periods
        signal_clarity = 1 - (hold_signals / total_periods)
        
        # 理想交易频率：10-30%
        if trade_frequency < 0.1 or trade_frequency > 0.3:
            frequency_score = 0
        else:
            frequency_score = 1 - abs(trade_frequency - 0.2)  # 距离理想值20%的偏差
        
        score = frequency_score * 0.6 + signal_clarity * 0.4
        return score

def test_signal_optimization():
    """测试信号优化"""
    print("测试信号阈值优化...")
    
    optimizer = SignalOptimizer()
    historical_data = generate_realistic_test_data(300)
    
    best_thresholds = optimizer.optimize_signal_thresholds(historical_data)
    
    print(f"\n📊 优化完成!")
    print(f"推荐使用阈值: {best_thresholds}")
    
    return best_thresholds

if __name__ == "__main__":
    test_signal_optimization()