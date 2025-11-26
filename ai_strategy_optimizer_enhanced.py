# ai_strategy_optimizer_enhanced.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from src.strategies.strategy_factory import strategy_factory

class EnhancedAIStrategyOptimizer:
    """增强版AI策略优化器 - 使用新策略架构"""
    
    def __init__(self):
        self.optimization_history = []
        self.best_strategies = {}
    
    def optimize_strategy_parameters(self, strategy_type: str, data: pd.DataFrame, 
                                   generations: int = 50, population_size: int = 20):
        """优化特定策略的参数"""
        print(f"🧬 开始优化 {strategy_type} 参数...")
        
        # 获取策略类和所需参数
        strategy_info = self._get_strategy_parameter_ranges(strategy_type)
        
        best_score = -np.inf
        best_params = None
        
        for generation in range(generations):
            population = self._generate_population(strategy_info, population_size)
            generation_scores = []
            
            for params in population:
                try:
                    # 使用新工厂创建策略
                    config = {
                        'name': f'优化_{strategy_type}',
                        'parameters': params
                    }
                    strategy = strategy_factory.create_strategy(strategy_type, config)
                    
                    # 评估策略
                    score = self._evaluate_strategy(strategy, data)
                    generation_scores.append((score, params))
                    
                    # 更新最佳结果
                    if score > best_score:
                        best_score = score
                        best_params = params
                        
                except Exception as e:
                    print(f"⚠️ 参数评估失败: {e}")
                    continue
            
            # 记录代际结果
            avg_score = np.mean([s[0] for s in generation_scores])
            self.optimization_history.append({
                'generation': generation,
                'best_score': best_score,
                'avg_score': avg_score,
                'strategy_type': strategy_type
            })
            
            if generation % 10 == 0:
                print(f"  第{generation}代: 最佳得分 = {best_score:.4f}, 平均得分 = {avg_score:.4f}")
        
        # 保存最佳策略
        self.best_strategies[strategy_type] = {
            'parameters': best_params,
            'score': best_score
        }
        
        print(f"🎯 {strategy_type} 优化完成!")
        print(f"   最佳参数: {best_params}")
        print(f"   最佳得分: {best_score:.4f}")
        
        return best_params, best_score
    
    def _get_strategy_parameter_ranges(self, strategy_type: str) -> Dict[str, Any]:
        """定义各策略的参数优化范围"""
        parameter_ranges = {
            'SimpleMovingAverageStrategy': {
                'sma_fast': {'min': 5, 'max': 50, 'type': 'int'},
                'sma_slow': {'min': 20, 'max': 100, 'type': 'int'}
            },
            'MACDStrategySmart': {
                'fast_period': {'min': 8, 'max': 20, 'type': 'int'},
                'slow_period': {'min': 20, 'max': 40, 'type': 'int'},
                'signal_period': {'min': 5, 'max': 15, 'type': 'int'}
            },
            'BollingerBandsStrategy': {
                'period': {'min': 10, 'max': 30, 'type': 'int'},
                'std_dev': {'min': 1.5, 'max': 3.0, 'type': 'float'}
            },
            'TurtleTradingStrategy': {
                'entry_period': {'min': 15, 'max': 30, 'type': 'int'},
                'exit_period': {'min': 5, 'max': 20, 'type': 'int'},
                'atr_period': {'min': 10, 'max': 20, 'type': 'int'}
            }
        }
        return parameter_ranges.get(strategy_type, {})
    
    def _generate_population(self, parameter_ranges: Dict, size: int) -> List[Dict]:
        """生成参数种群"""
        population = []
        for _ in range(size):
            individual = {}
            for param, ranges in parameter_ranges.items():
                if ranges['type'] == 'int':
                    individual[param] = np.random.randint(ranges['min'], ranges['max'])
                else:  # float
                    individual[param] = np.random.uniform(ranges['min'], ranges['max'])
            population.append(individual)
        return population
    
    def _evaluate_strategy(self, strategy, data: pd.DataFrame) -> float:
        """评估策略性能"""
        try:
            signals = strategy.calculate_signals(data)
            if signals.empty:
                return -np.inf
            
            # 简单的性能评估：信号变化频率和幅度
            signal_changes = (signals['signal'].diff() != 0).sum()
            signal_strength = signals['signal'].abs().mean()
            
            # 组合评分（需要根据实际需求调整）
            score = signal_changes * 0.1 + signal_strength * 0.9
            return score
            
        except Exception as e:
            return -np.inf

def test_enhanced_optimizer():
    """测试增强版优化器"""
    print("🚀 测试增强版AI策略优化器...")
    
    # 生成测试数据
    from test_strategies_with_real_data import generate_realistic_test_data
    test_data = generate_realistic_test_data(200)  # 更长的数据用于优化
    
    optimizer = EnhancedAIStrategyOptimizer()
    
    # 优化SMA策略
    best_params, best_score = optimizer.optimize_strategy_parameters(
        'SimpleMovingAverageStrategy', test_data, generations=20, population_size=10
    )
    
    print(f"\n📊 优化历史长度: {len(optimizer.optimization_history)}")
    print(f"💾 最佳策略保存: {list(optimizer.best_strategies.keys())}")
    
    return optimizer

if __name__ == "__main__":
    test_enhanced_optimizer()