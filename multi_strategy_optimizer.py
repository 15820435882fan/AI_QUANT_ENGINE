# multi_strategy_optimizer.py
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from multi_strategy_manager_enhanced import MultiStrategyManagerEnhanced

class MultiStrategyOptimizer:
    """多策略组合优化器"""
    
    def __init__(self):
        self.manager = MultiStrategyManagerEnhanced()
        self.optimization_results = {}
    
    def optimize_strategy_combination(self, data: pd.DataFrame, 
                                    strategy_combinations: List[List[str]]):
        """优化策略组合"""
        print("🧬 开始多策略组合优化...")
        
        best_combination = None
        best_score = -np.inf
        
        for i, combination in enumerate(strategy_combinations):
            print(f"\n🔍 测试组合 {i+1}/{len(strategy_combinations)}: {combination}")
            
            # 清空当前策略
            self.manager.strategies.clear()
            
            # 添加策略组合
            for strategy_type in combination:
                config = self._get_default_config(strategy_type)
                self.manager.add_strategy(strategy_type, config)
            
            # 评估组合
            score = self._evaluate_combination(data)
            
            if score > best_score:
                best_score = score
                best_combination = combination
            
            self.optimization_results[str(tuple(combination))] = score
        
        print(f"\n🎯 最佳策略组合: {best_combination}")
        print(f"📊 最佳得分: {best_score:.4f}")
        
        return best_combination, best_score
    
    def _get_default_config(self, strategy_type: str) -> Dict:
        """获取策略的默认配置"""
        default_configs = {
            'SimpleMovingAverageStrategy': {
                'name': f'{strategy_type}_默认',
                'parameters': {'sma_fast': 10, 'sma_slow': 30}
            },
            'MACDStrategySmart': {
                'name': f'{strategy_type}_默认', 
                'parameters': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
            },
            'BollingerBandsStrategy': {
                'name': f'{strategy_type}_默认',
                'parameters': {'period': 20, 'std_dev': 2.0}
            }
        }
        return default_configs.get(strategy_type, {'name': strategy_type, 'parameters': {}})
    
    def _evaluate_combination(self, data: pd.DataFrame) -> float:
        """评估策略组合性能"""
        try:
            combined_signals = self.manager.calculate_combined_signals(data)
            if combined_signals.empty:
                return -np.inf
            
            # 计算组合信号的质量
            signal_variance = combined_signals['combined_signal'].var()
            signal_changes = (combined_signals['combined_signal'].diff() != 0).sum()
            
            # 组合评分（信号稳定性 + 适当的变化频率）
            score = signal_variance * 0.7 + min(signal_changes, 10) * 0.3
            return score
            
        except Exception as e:
            print(f"⚠️ 组合评估失败: {e}")
            return -np.inf

def test_multi_strategy_optimizer():
    """测试多策略优化器"""
    print("🚀 测试多策略组合优化器...")
    
    # 生成测试数据
    from test_strategies_with_real_data import generate_realistic_test_data
    test_data = generate_realistic_test_data(150)
    
    optimizer = MultiStrategyOptimizer()
    
    # 定义要测试的策略组合
    strategy_combinations = [
        ['SimpleMovingAverageStrategy', 'MACDStrategySmart'],
        ['SimpleMovingAverageStrategy', 'BollingerBandsStrategy'],
        ['MACDStrategySmart', 'BollingerBandsStrategy'],
        ['SimpleMovingAverageStrategy', 'MACDStrategySmart', 'BollingerBandsStrategy']
    ]
    
    best_combination, best_score = optimizer.optimize_strategy_combination(
        test_data, strategy_combinations
    )
    
    print(f"\n📊 所有组合结果:")
    for combo, score in optimizer.optimization_results.items():
        print(f"  {combo}: {score:.4f}")
    
    return optimizer

if __name__ == "__main__":
    test_multi_strategy_optimizer()