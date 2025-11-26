# src/strategies/risk_enhanced_engine.py
class RiskEnhancedEngine:
    """增强风险控制引擎"""
    
    def __init__(self):
        self.stop_loss_pct = 0.08  # 8%止损
        self.take_profit_pct = 0.15  # 15%止盈
        self.trailing_stop_pct = 0.05  # 5%移动止损
        self.max_position_hold_days = 30  # 最大持仓30天
        
    def check_stop_loss(self, current_price: float, entry_price: float, 
                       position_type: str) -> Tuple[bool, str]:
        """检查止损条件"""
        if position_type == 'LONG':
            pnl_pct = (current_price - entry_price) / entry_price
            if pnl_pct <= -self.stop_loss_pct:
                return True, f"止损触发: 亏损{pnl_pct:.2%}"
            elif pnl_pct >= self.take_profit_pct:
                return True, f"止盈触发: 盈利{pnl_pct:.2%}"
        else:  # SHORT
            pnl_pct = (entry_price - current_price) / entry_price
            if pnl_pct <= -self.stop_loss_pct:
                return True, f"止损触发: 亏损{pnl_pct:.2%}"
            elif pnl_pct >= self.take_profit_pct:
                return True, f"止盈触发: 盈利{pnl_pct:.2%}"
                
        return False, ""