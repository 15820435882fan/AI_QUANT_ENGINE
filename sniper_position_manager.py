# sniper_position_manager.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple  # 添加导入

class SniperPositionManager:
    """刺客仓位管理系统 - 修复版"""
    
    def __init__(self, total_capital: float = 10000.0):
        self.total_capital = total_capital
        self.used_capital = 0.0
        self.max_position_value = 1000  # 单次最大建仓
        self.min_position_value = 100   # 单次最小建仓
        self.stop_loss_pct = 0.02       # 2%止损
        self.take_profit_pct = 0.06     # 6%止盈
        self.max_drawdown = 0.10        # 最大回撤10%
        
    def calculate_position_size(self, signal: Dict) -> Dict[str, Any]:
        """计算刺客仓位"""
        base_capital = min(self.total_capital * 0.1, self.max_position_value)  # 最多使用10%资金
        
        # 根据信号强度调整仓位
        confidence = signal.get('confidence', 0.5)
        if confidence > 0.9:
            position_size = base_capital
        elif confidence > 0.8:
            position_size = base_capital * 0.7
        elif confidence > 0.7:
            position_size = base_capital * 0.5
        else:
            position_size = base_capital * 0.3
        
        # 确保在最小最大范围内
        position_size = max(self.min_position_value, min(position_size, self.max_position_value))
        
        # 杠杆计算
        leverage = signal.get('leverage', 10)
        entry_price = signal.get('entry_price', 1)
        notional_value = position_size * leverage
        
        return {
            'position_size': position_size,
            'leverage': leverage,
            'notional_value': notional_value,
            'quantity': position_size / entry_price if entry_price > 0 else 0,
            'stop_loss': self._calculate_stop_loss(signal),
            'take_profit': self._calculate_take_profit(signal)
        }
    
    def _calculate_stop_loss(self, signal: Dict) -> float:
        """计算止损价格"""
        entry_price = signal.get('entry_price', 1)
        direction = signal.get('direction', 'LONG')
        
        if direction == 'LONG':
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)
    
    def _calculate_take_profit(self, signal: Dict) -> float:
        """计算止盈价格"""
        entry_price = signal.get('entry_price', 1)
        direction = signal.get('direction', 'LONG')
        
        if direction == 'LONG':
            return entry_price * (1 + self.take_profit_pct)
        else:
            return entry_price * (1 - self.take_profit_pct)

# 测试函数
def test_position_manager():
    """测试仓位管理器"""
    print("🧪 测试仓位管理器...")
    
    manager = SniperPositionManager(10000.0)
    
    # 测试信号
    test_signal = {
        'direction': 'LONG',
        'confidence': 0.85,
        'leverage': 10,
        'entry_price': 50000.0
    }
    
    position = manager.calculate_position_size(test_signal)
    
    print(f"📊 仓位计算结果:")
    print(f"  仓位大小: ${position['position_size']:.2f}")
    print(f"  杠杆: {position['leverage']}x")
    print(f"  名义价值: ${position['notional_value']:.2f}")
    print(f"  数量: {position['quantity']:.6f}")
    print(f"  止损: ${position['stop_loss']:.2f}")
    print(f"  止盈: ${position['take_profit']:.2f}")
    
    return manager

if __name__ == "__main__":
    test_position_manager()