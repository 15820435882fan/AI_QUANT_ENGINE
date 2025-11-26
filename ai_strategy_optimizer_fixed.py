# ai_strategy_optimizer_fixed.py
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from typing import Dict, List
import asyncio

class AIStrategyOptimizerFixed:
    """修复版AI策略优化器"""
    
    def __init__(self, population_size: int = 20, generations: int = 30):
        self.population_size = population_size
        self.generations = generations
        
        # 简化的策略参数空间
        self.strategy_spaces = {
            'momentum': {
                'lookback': (5, 20),  # 观察期
                'threshold': (0.001, 0.01)  # 阈值
            },
            'reversion': {
                'lookback': (10, 30),
                'deviation': (0.005, 0.02)
            }
        }
        
        self.population = []
        self.best_score_history = []
    
    def initialize_population(self):
        """初始化种群"""
        self.population = []
        strategy_types = list(self.strategy_spaces.keys())
        
        for _ in range(self.population_size):
            strategy_type = np.random.choice(strategy_types)
            parameters = {}
            
            for param, (min_val, max_val) in self.strategy_spaces[strategy_type].items():
                parameters[param] = np.random.uniform(min_val, max_val)
            
            self.population.append({
                'type': strategy_type,
                'parameters': parameters,
                'score': 0.0
            })
    
    def simulate_strategy_fixed(self, strategy, data: pd.DataFrame) -> List[float]:
        """修复版策略模拟"""
        returns = []
        position = 0
        entry_price = 0
        
        closes = data['close'].values
        
        for i in range(1, len(closes)):
            current_price = closes[i]
            prev_price = closes[i-1]
            
            if strategy['type'] == 'momentum':
                # 动量策略
                lookback = max(1, int(strategy['parameters']['lookback']))
                threshold = strategy['parameters']['threshold']
                
                if i > lookback:
                    past_return = (current_price - closes[i-lookback]) / closes[i-lookback]
                    
                    if past_return > threshold and position <= 0:
                        if position == -1:  # 平空仓
                            trade_return = (entry_price - current_price) / entry_price
                            returns.append(trade_return)
                        position = 1  # 开多仓
                        entry_price = current_price
                    
                    elif past_return < -threshold and position >= 0:
                        if position == 1:  # 平多仓
                            trade_return = (current_price - entry_price) / entry_price
                            returns.append(trade_return)
                        position = -1  # 开空仓
                        entry_price = current_price
            
            elif strategy['type'] == 'reversion':
                # 均值回归策略
                lookback = max(5, int(strategy['parameters']['lookback']))
                deviation = strategy['parameters']['deviation']
                
                if i > lookback:
                    ma = np.mean(closes[i-lookback:i])
                    current_deviation = (current_price - ma) / ma
                    
                    if current_deviation < -deviation and position <= 0:  # 超卖
                        if position == -1:
                            trade_return = (entry_price - current_price) / entry_price
                            returns.append(trade_return)
                        position = 1
                        entry_price = current_price
                    
                    elif current_deviation > deviation and position >= 0:  # 超买
                        if position == 1:
                            trade_return = (current_price - entry_price) / entry_price
                            returns.append(trade_return)
                        position = -1
                        entry_price = current_price
        
        # 平最后仓位
        if position != 0:
            if position == 1:
                final_return = (closes[-1] - entry_price) / entry_price
            else:
                final_return = (entry_price - closes[-1]) / entry_price
            returns.append(final_return)
        
        return returns
    
    def evaluate_strategy(self, strategy, data: pd.DataFrame) -> float:
        """评估策略"""
        try:
            returns = self.simulate_strategy_fixed(strategy, data)
            
            if len(returns) < 3:  # 交易次数太少
                return -5.0
            
            # 计算性能指标
            total_return = np.sum(returns)
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            # 综合得分
            score = total_return * 0.6 + sharpe * 0.4
            
            return max(score, -10.0)
            
        except Exception as e:
            print(f"评估错误: {e}")
            return -10.0
    
    async def evolve_population(self, data: pd.DataFrame):
        """进化种群"""
        # 评估所有策略
        for strategy in self.population:
            strategy['score'] = self.evaluate_strategy(strategy, data)
        
        # 选择
        scores = [s['score'] for s in self.population]
        best_idx = np.argmax(scores)
        self.best_score_history.append(scores[best_idx])
        
        # 创建新种群
        new_population = [self.population[best_idx]]  # 保留最佳
        
        for _ in range(self.population_size - 1):
            # 轮盘赌选择
            exp_scores = np.exp(scores - np.max(scores))
            probabilities = exp_scores / np.sum(exp_scores)
            parent_idx = np.random.choice(len(self.population), p=probabilities)
            parent = self.population[parent_idx]
            
            # 变异
            child = self.mutate(parent)
            new_population.append(child)
        
        self.population = new_population
    
    def mutate(self, strategy, mutation_rate: float = 0.3):
        """变异"""
        child = {
            'type': strategy['type'],
            'parameters': strategy['parameters'].copy(),
            'score': 0.0
        }
        
        if np.random.random() < mutation_rate:
            param_to_mutate = np.random.choice(list(child['parameters'].keys()))
            min_val, max_val = self.strategy_spaces[child['type']][param_to_mutate]
            child['parameters'][param_to_mutate] = np.random.uniform(min_val, max_val)
        
        return child
    
    async def optimize(self, data: pd.DataFrame):
        """执行优化"""
        print("🧬 开始AI策略优化...")
        self.initialize_population()
        
        for generation in range(self.generations):
            await self.evolve_population(data)
            
            if generation % 5 == 0:
                best_score = max(s['score'] for s in self.population)
                print(f"   第{generation}代: 最佳得分 = {best_score:.4f}")
        
        # 返回最佳策略
        best_strategy = max(self.population, key=lambda x: x['score'])
        print(f"🎯 优化完成!")
        print(f"   最佳策略类型: {best_strategy['type']}")
        print(f"   参数: {best_strategy['parameters']}")
        print(f"   最终得分: {best_strategy['score']:.4f}")
        
        return best_strategy

async def test_ai_optimizer_fixed():
    """测试修复版优化器"""
    # 生成更好的测试数据
    np.random.seed(42)
    n_points = 1000  # 更多数据点
    
    # 创建有趋势的数据
    prices = [50000]
    trend = 0.0001  # 轻微上升趋势
    
    for i in range(1, n_points):
        # 趋势 + 噪声 + 周期性
        noise = np.random.normal(0, 0.005)
        cycle = 0.002 * np.sin(i * 0.01)
        
        change = trend + noise + cycle
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 100))
    
    data = pd.DataFrame({
        'close': prices,
        'high': [p * 1.005 for p in prices],
        'low': [p * 0.995 for p in prices]
    })
    
    optimizer = AIStrategyOptimizerFixed(population_size=15, generations=20)
    best_strategy = await optimizer.optimize(data)
    
    return best_strategy

if __name__ == "__main__":
    best = asyncio.run(test_ai_optimizer_fixed())