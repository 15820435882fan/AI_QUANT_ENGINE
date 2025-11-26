# ai_strategy_optimizer.py
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class StrategyGene:
    """策略基因 - 用于遗传算法优化"""
    strategy_type: str
    parameters: Dict
    performance: float = 0.0
    weight: float = 0.0

class AIStrategyOptimizer:
    """AI策略优化器 - 使用遗传算法优化策略组合"""
    
    def __init__(self, population_size: int = 50, generations: int = 100):
        self.population_size = population_size
        self.generations = generations
        self.population: List[StrategyGene] = []
        self.best_strategies: List[StrategyGene] = []
        
        # 定义策略参数空间
        self.strategy_spaces = {
            'rsi': {
                'rsi_period': (5, 21),
                'oversold': (20, 40),
                'overbought': (60, 80)
            },
            'ema': {
                'fast_period': (3, 12),
                'slow_period': (15, 30)
            },
            'macd': {
                'fast_period': (6, 12),
                'slow_period': (18, 26),
                'signal_period': (5, 9)
            },
            'bollinger': {
                'period': (10, 20),
                'std_dev': (1.5, 2.5)
            }
        }
    
    def initialize_population(self):
        """初始化策略种群"""
        self.population = []
        
        for _ in range(self.population_size):
            strategy_type = np.random.choice(list(self.strategy_spaces.keys()))
            parameters = {}
            
            for param, (min_val, max_val) in self.strategy_spaces[strategy_type].items():
                if isinstance(min_val, int):
                    parameters[param] = np.random.randint(min_val, max_val + 1)
                else:
                    parameters[param] = np.random.uniform(min_val, max_val)
            
            self.population.append(StrategyGene(strategy_type, parameters))
    
    async def evaluate_strategy(self, gene: StrategyGene, historical_data: pd.DataFrame) -> float:
        """评估策略性能"""
        try:
            # 模拟策略回测
            returns = self.simulate_strategy(gene, historical_data)
            
            if len(returns) == 0:
                return -10.0  # 惩罚无交易策略
                
            # 综合评估指标
            total_return = np.prod([1 + r for r in returns]) - 1
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            max_drawdown = self.calculate_max_drawdown(returns)
            
            # 综合得分
            score = (total_return * 0.4 + 
                    sharpe_ratio * 0.3 + 
                    (1 - max_drawdown) * 0.3)
            
            return max(score, -10.0)
            
        except Exception as e:
            print(f"策略评估错误: {e}")
            return -10.0
    
    def simulate_strategy(self, gene: StrategyGene, data: pd.DataFrame) -> List[float]:
        """简化版策略模拟"""
        returns = []
        position = 0
        entry_price = 0
        
        for i in range(1, len(data)):
            current_price = data['close'].iloc[i]
            prev_price = data['close'].iloc[i-1]
            
            # 简化交易逻辑
            price_change = (current_price - prev_price) / prev_price
            
            if gene.strategy_type == 'rsi':
                # RSI策略逻辑
                if price_change < -0.02 and position <= 0:  # 超卖信号
                    position = 1
                    entry_price = current_price
                elif price_change > 0.02 and position >= 0:  # 超买信号
                    if position == 1:
                        returns.append((current_price - entry_price) / entry_price)
                    position = -1
                    entry_price = current_price
                    
            elif gene.strategy_type == 'ema':
                # EMA策略逻辑
                if i > gene.parameters.get('slow_period', 20):
                    ema_fast = data['close'].iloc[i-gene.parameters['fast_period']:i].mean()
                    ema_slow = data['close'].iloc[i-gene.parameters['slow_period']:i].mean()
                    
                    if ema_fast > ema_slow and position <= 0:
                        if position == -1:
                            returns.append((current_price - entry_price) / entry_price)
                        position = 1
                        entry_price = current_price
                    elif ema_fast < ema_slow and position >= 0:
                        if position == 1:
                            returns.append((current_price - entry_price) / entry_price)
                        position = -1
                        entry_price = current_price
        
        return returns
    
    def calculate_max_drawdown(self, returns: List[float]) -> float:
        """计算最大回撤"""
        cumulative = np.cumprod([1 + r for r in returns])
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        return np.max(drawdown) if len(drawdown) > 0 else 0
    
    async def evolve_population(self, historical_data: pd.DataFrame):
        """进化策略种群"""
        # 评估所有策略
        evaluations = []
        for gene in self.population:
            score = await self.evaluate_strategy(gene, historical_data)
            gene.performance = score
            evaluations.append(score)
        
        # 选择优秀策略
        evaluations = np.array(evaluations)
        probabilities = np.exp(evaluations - np.max(evaluations))  # Softmax选择
        probabilities /= probabilities.sum()
        
        # 选择和交叉
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = np.random.choice(self.population, size=2, p=probabilities)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        
        self.population = new_population
        self.best_strategies.append(self.population[np.argmax(evaluations)])
    
    def crossover(self, parent1: StrategyGene, parent2: StrategyGene) -> StrategyGene:
        """策略交叉"""
        if parent1.strategy_type == parent2.strategy_type:
            # 同类型策略参数交叉
            child_params = {}
            for param in parent1.parameters.keys():
                if np.random.random() < 0.5:
                    child_params[param] = parent1.parameters[param]
                else:
                    child_params[param] = parent2.parameters[param]
            return StrategyGene(parent1.strategy_type, child_params)
        else:
            # 不同类型策略，随机选择
            return parent1 if np.random.random() < 0.5 else parent2
    
    def mutate(self, gene: StrategyGene, mutation_rate: float = 0.1) -> StrategyGene:
        """策略变异"""
        if np.random.random() < mutation_rate:
            param_to_mutate = np.random.choice(list(gene.parameters.keys()))
            min_val, max_val = self.strategy_spaces[gene.strategy_type][param_to_mutate]
            
            if isinstance(min_val, int):
                gene.parameters[param_to_mutate] = np.random.randint(min_val, max_val + 1)
            else:
                gene.parameters[param_to_mutate] = np.random.uniform(min_val, max_val)
        
        return gene
    
    async def optimize(self, historical_data: pd.DataFrame):
        """执行优化过程"""
        print("🧬 开始AI策略优化...")
        self.initialize_population()
        
        for generation in range(self.generations):
            await self.evolve_population(historical_data)
            
            if generation % 10 == 0:
                best_score = max(gene.performance for gene in self.population)
                print(f"   Generation {generation}: Best Score = {best_score:.4f}")
        
        # 选择最终最佳策略
        best_gene = max(self.population, key=lambda x: x.performance)
        print(f"🎯 优化完成! 最佳策略: {best_gene.strategy_type}")
        print(f"   参数: {best_gene.parameters}")
        print(f"   性能得分: {best_gene.performance:.4f}")
        
        return best_gene

# 测试优化器
async def test_ai_optimizer():
    """测试AI策略优化器"""
    # 生成模拟历史数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='5min')
    prices = [50000]
    
    for i in range(1, len(dates)):
        change = np.random.normal(0, 0.001)  # 小幅波动
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    historical_data = pd.DataFrame({
        'timestamp': dates,
        'close': prices
    })
    
    optimizer = AIStrategyOptimizer(population_size=20, generations=50)
    best_strategy = await optimizer.optimize(historical_data)
    
    return best_strategy

if __name__ == "__main__":
    best = asyncio.run(test_ai_optimizer())