# dynamic_capital_manager.py
import pandas as pd
from typing import Dict, Any, List
from decimal import Decimal, ROUND_DOWN

class DynamicCapitalManager:
    """动态资金管理器 - 支持资金划转监控"""
    
    def __init__(self, total_capital: float = 10000.0):
        self.total_capital = total_capital
        self.available_capital = total_capital
        self.used_capital = 0.0
        self.active_positions = {}
        self.capital_history = []
        
        # 资金分配策略
        self.small_capital_threshold = 1000  # 小资金阈值
        self.small_position_ratio = 0.02     # 小资金仓位比例
        self.large_position_ratio = 0.08     # 大资金仓位比例
        self.max_positions = 10              # 最大同时持仓数
    
    def update_account_balance(self, new_balance: float):
        """更新账户余额（支持资金划转）"""
        balance_change = new_balance - self.total_capital
        self.total_capital = new_balance
        self.available_capital += balance_change
        
        # 记录资金变动
        self.capital_history.append({
            'timestamp': pd.Timestamp.now(),
            'total_capital': self.total_capital,
            'available_capital': self.available_capital,
            'balance_change': balance_change
        })
    
    def calculate_position_size(self, symbol: str, signal: Dict, is_main_symbol: bool = False) -> Dict[str, Any]:
        """计算动态仓位大小"""
        # 检查是否已有该币种仓位
        if symbol in self.active_positions:
            return {'position_size': 0, 'error': '已有该币种仓位'}
        
        # 检查最大持仓限制
        if len(self.active_positions) >= self.max_positions:
            return {'position_size': 0, 'error': '达到最大持仓限制'}
        
        # 根据资金规模确定仓位比例
        if self.total_capital <= self.small_capital_threshold:
            position_ratio = self.small_position_ratio
            min_position = 50  # 最小50USDT
        else:
            position_ratio = self.small_position_ratio if not is_main_symbol else self.large_position_ratio
            min_position = 100  # 最小100USDT
        
        # 计算仓位大小
        base_size = self.total_capital * position_ratio
        confidence = signal.get('confidence', 0.5)
        adjusted_size = base_size * confidence
        
        # 应用限制
        position_size = min(adjusted_size, self.available_capital * 0.8)  # 不超过可用资金的80%
        position_size = max(position_size, min_position)
        
        # 确保有足够资金
        if position_size > self.available_capital:
            position_size = self.available_capital * 0.9
        
        return {
            'position_size': position_size,
            'leverage': signal.get('leverage', 10),
            'quantity': position_size / signal['entry_price'],
            'available_capital_before': self.available_capital,
            'is_main_symbol': is_main_symbol
        }
    
    def open_position(self, symbol: str, position_info: Dict):
        """开仓"""
        position_size = position_info['position_size']
        
        if position_size <= self.available_capital:
            self.active_positions[symbol] = position_info
            self.used_capital += position_size
            self.available_capital -= position_size
            
            print(f"✅ 开仓 {symbol}: ${position_size:.0f}, 可用资金: ${self.available_capital:.0f}")
    
    def close_position(self, symbol: str, pnl: float):
        """平仓"""
        if symbol in self.active_positions:
            position = self.active_positions[symbol]
            position_size = position['position_size']
            
            self.used_capital -= position_size
            self.available_capital += position_size + pnl
            
            # 更新总资金
            self.total_capital += pnl
            
            del self.active_positions[symbol]
            
            status = "盈利" if pnl > 0 else "亏损"
            print(f"🔚 平仓 {symbol}: {status} ${pnl:+.0f}, 总资金: ${self.total_capital:.0f}")
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """获取投资组合状态"""
        total_position_value = sum(pos['position_size'] for pos in self.active_positions.values())
        
        return {
            'total_capital': self.total_capital,
            'available_capital': self.available_capital,
            'used_capital': total_position_value,
            'active_positions': len(self.active_positions),
            'utilization_rate': total_position_value / self.total_capital if self.total_capital > 0 else 0,
            'max_positions': self.max_positions
        }