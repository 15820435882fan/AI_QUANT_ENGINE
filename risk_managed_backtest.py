# risk_managed_backtest.py
class RiskManagedBacktest:
    def __init__(self, initial_capital=10000, max_drawdown=0.15, daily_loss_limit=0.03):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_drawdown = max_drawdown
        self.daily_loss_limit = daily_loss_limit
        self.daily_pnl = 0
        self.trades = []
        
        self.peak_capital = initial_capital
        
    def risk_checks(self, proposed_trade_size):
        """风险检查"""
        # 最大回撤检查
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_drawdown > self.max_drawdown:
            return False, "超过最大回撤限制"
        
        # 单日亏损限制
        if self.daily_pnl < -self.daily_loss_limit * self.initial_capital:
            return False, "超过单日亏损限制"
        
        # 仓位大小限制
        if proposed_trade_size > self.current_capital * 0.3:
            return False, "仓位过大"
            
        return True, "通过"
    
    def execute_risk_managed_trade(self, signal, price_data):
        """执行风险管理的交易"""
        # 计算动态仓位
        strategy = EnhancedTradingStrategy()
        position_size = strategy.dynamic_position_size(
            signal['strength'], 
            [t['profit_pct'] for t in self.trades[-10:]]
        )
        
        # 风险检查
        trade_amount = self.current_capital * position_size
        risk_ok, risk_msg = self.risk_checks(trade_amount)
        
        if not risk_ok:
            return {'executed': False, 'reason': risk_msg}
        
        # 执行交易 (模拟)
        # ... 交易逻辑
        
        # 更新资金和风险指标
        self.current_capital += profit
        self.daily_pnl += profit
        self.peak_capital = max(self.peak_capital, self.current_capital)
        
        return {'executed': True, 'profit': profit}