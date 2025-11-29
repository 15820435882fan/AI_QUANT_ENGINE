# strategy_optimizer_v1.py
import numpy as np

class PositionOptimizer:
    def __init__(self):
        self.optimization_history = []
    
    def optimize_position_size(self, base_win_rate, avg_profit, avg_loss, current_capital):
        """基于凯利公式优化仓位"""
        win_rate = base_win_rate
        win_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 1
        
        # 凯利公式: f = (bp - q) / b
        b = win_ratio  # 盈亏比
        p = win_rate   # 胜率
        q = 1 - p      # 败率
        
        kelly_fraction = (b * p - q) / b if b > 0 else 0
        # 保守版本：使用半凯利
        optimal_position = max(0.05, min(kelly_fraction * 0.5, 0.3))
        
        return optimal_position

# 基于你的回测结果优化
optimizer = PositionOptimizer()
optimal_size = optimizer.optimize_position_size(
    base_win_rate=0.738,
    avg_profit=2.01, 
    avg_loss=-1.5,  # 估计值
    current_capital=10322
)
print(f"优化后仓位比例: {optimal_size:.1%}")