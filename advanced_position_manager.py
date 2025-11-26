# advanced_position_manager.py
import pandas as pd
import numpy as np
from typing import Dict, Any

class AdvancedPositionManager:
    """优化版高级仓位管理器 - 保守风险控制"""
    
    def __init__(self, total_capital: float = 10000.0):
        self.total_capital = total_capital
        self.used_capital = 0.0
        self.active_positions = {}
        
        # 🎯 优化仓位参数 - 更保守
        self.base_position_size = 0.03    # 降低基础仓位到3%
        self.max_position_size = 0.12     # 降低最大仓位到12%
        self.risk_per_trade = 0.01        # 降低单笔风险到1%
        
        # 🎯 优化止损止盈 - 改善风险回报比
        self.base_stop_loss_pct = 0.035   # 放宽止损到3.5%
        self.base_take_profit_pct = 0.09  # 提高止盈到9%
        
        # 🎯 杠杆控制
        self.base_leverage = 6            # 降低基础杠杆
        self.max_leverage = 10            # 降低最大杠杆
        
    def calculate_dynamic_position(self, signal: Dict, market_conditions: Dict) -> Dict[str, Any]:
        """计算动态仓位 - 更保守的风险控制"""
        # 基础置信度
        confidence = signal.get('confidence', 0.5)
        
        # 🎯 优化波动率调整 - 更合理的逻辑
        volatility = market_conditions.get('volatility', 0.02)
        
        # 波动率越高，仓位应该越小，杠杆越低
        if volatility < 0.01:
            vol_adjustment = 1.3  # 低波动率，稍微增加仓位
        elif volatility < 0.02:
            vol_adjustment = 1.0  # 正常波动率
        elif volatility < 0.03:
            vol_adjustment = 0.7  # 较高波动率，减少仓位
        else:
            vol_adjustment = 0.5  # 高波动率，大幅减少仓位
        
        # 🎯 优化信号强度调整
        technical_score = signal.get('technical_score', {})
        if isinstance(technical_score, dict):
            trend_strength = abs(technical_score.get('trend_strength', 0))
            momentum = abs(technical_score.get('momentum', 0))
            # 综合信号强度
            signal_strength = min(1.0, (trend_strength * 0.6 + min(abs(momentum) * 10, 0.4)))
        else:
            signal_strength = 0.5
            
        signal_adjustment = 0.7 + signal_strength * 0.6  # 0.7-1.3范围
        
        # 🎯 计算基础仓位 - 更保守
        base_size = self.total_capital * self.base_position_size
        adjusted_size = base_size * confidence * vol_adjustment * signal_adjustment
        
        # 应用限制
        position_size = min(adjusted_size, self.total_capital * self.max_position_size)
        position_size = max(position_size, 100)  # 最小100美元
        
        # 🎯 优化杠杆计算 - 基于波动率和置信度
        leverage_ratio = (0.02 / max(volatility, 0.01)) * confidence
        leverage = int(max(3, min(self.base_leverage * leverage_ratio, self.max_leverage)))
        
        # 🎯 动态止损止盈 - 基于波动率调整
        stop_loss_pct = self.base_stop_loss_pct * (volatility / 0.02)  # 波动率越高，止损越宽
        take_profit_pct = self.base_take_profit_pct * (0.02 / max(volatility, 0.01))  # 波动率越低，止盈越紧
        
        # 确保风险回报比至少1:2
        min_take_profit = stop_loss_pct * 2.5
        take_profit_pct = max(take_profit_pct, min_take_profit)
        
        # 限制范围
        stop_loss_pct = min(max(stop_loss_pct, 0.025), 0.06)    # 2.5%-6%
        take_profit_pct = min(max(take_profit_pct, 0.06), 0.15) # 6%-15%
        
        entry_price = signal.get('entry_price', 1)
        direction = signal.get('direction', 'LONG')
        
        if direction == 'LONG':
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
        else:  # SHORT
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)
        
        # 🎯 计算实际风险
        actual_risk_pct = (abs(entry_price - stop_loss) / entry_price) * leverage
        risk_amount = position_size * actual_risk_pct
        
        return {
            'position_size': position_size,
            'leverage': leverage,
            'notional_value': position_size * leverage,
            'quantity': position_size / entry_price if entry_price > 0 else 0,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'risk_amount': risk_amount,
            'actual_risk_pct': actual_risk_pct,
            'confidence': confidence,
            'volatility_adjustment': vol_adjustment,
            'signal_adjustment': signal_adjustment
        }
    
    def can_open_new_position(self, symbol: str) -> bool:
        """检查是否可以开新仓位"""
        # 检查是否已有该币种的仓位
        if symbol in self.active_positions:
            return False
        
        # 检查总仓位限制
        total_used = sum(pos['position_size'] for pos in self.active_positions.values())
        available_capital = self.total_capital - total_used
        
        return available_capital >= self.total_capital * self.base_position_size
    
    def add_position(self, symbol: str, position_info: Dict):
        """添加仓位记录"""
        self.active_positions[symbol] = position_info
        self.used_capital += position_info['position_size']
    
    def remove_position(self, symbol: str):
        """移除仓位记录"""
        if symbol in self.active_positions:
            position = self.active_positions[symbol]
            self.used_capital -= position['position_size']
            del self.active_positions[symbol]
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """获取投资组合状态"""
        total_position_value = sum(pos['position_size'] for pos in self.active_positions.values())
        available_capital = self.total_capital - total_position_value
        
        return {
            'total_capital': self.total_capital,
            'used_capital': total_position_value,
            'available_capital': available_capital,
            'position_count': len(self.active_positions),
            'utilization_rate': total_position_value / self.total_capital
        }

# 测试函数
def test_advanced_position_manager():
    """测试优化版高级仓位管理器"""
    print("🧪 测试优化版高级仓位管理器...")
    
    manager = AdvancedPositionManager(10000.0)
    
    # 测试信号 - 高置信度
    test_signal = {
        'direction': 'LONG',
        'confidence': 0.85,
        'entry_price': 50000.0,
        'technical_score': {
            'trend_strength': 0.8,
            'momentum': 0.03,
            'volatility': 0.015,
            'rsi': 65
        }
    }
    
    # 不同市场条件测试
    market_conditions_list = [
        {'volatility': 0.01, 'name': '低波动'},
        {'volatility': 0.02, 'name': '正常波动'}, 
        {'volatility': 0.04, 'name': '高波动'}
    ]
    
    for market_conditions in market_conditions_list:
        print(f"\n📊 {market_conditions['name']}市场条件:")
        position = manager.calculate_dynamic_position(test_signal, market_conditions)
        
        print(f"  仓位大小: ${position['position_size']:.0f}")
        print(f"  杠杆: {position['leverage']}x")
        print(f"  止损: {position['stop_loss_pct']:.1%}")
        print(f"  止盈: {position['take_profit_pct']:.1%}")
        print(f"  风险回报比: 1:{position['take_profit_pct']/position['stop_loss_pct']:.1f}")
        print(f"  实际风险: {position['actual_risk_pct']:.1%}")
        print(f"  风险金额: ${position['risk_amount']:.0f}")
    
    # 测试投资组合状态
    portfolio_status = manager.get_portfolio_status()
    print(f"\n📈 投资组合状态:")
    print(f"  总资金: ${portfolio_status['total_capital']:.0f}")
    print(f"  最大单仓位: ${portfolio_status['total_capital'] * 0.12:.0f}")
    print(f"  单笔最大风险: ${portfolio_status['total_capital'] * 0.01:.0f}")
    print(f"  杠杆范围: 3-{manager.max_leverage}x")
    
    return manager

if __name__ == "__main__":
    test_advanced_position_manager()