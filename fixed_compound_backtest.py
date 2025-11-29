# fixed_compound_backtest.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class FixedCompoundBacktest:
    def __init__(self, initial_capital=10000, position_size=0.1, use_compound=True):
        self.initial_capital = initial_capital
        self.position_size = position_size  # 固定仓位比例
        self.use_compound = use_compound
        self.current_capital = initial_capital
        self.trades = []
        self.portfolio_values = []
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def calculate_risk_metrics(self, returns):
        """计算风险评估指标"""
        if len(returns) == 0:
            return {
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'volatility': 0,
                'calmar_ratio': 0
            }
        
        # 夏普比率 (假设无风险利率为0)
        excess_returns = returns - 0
        sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(365) if np.std(returns) > 0 else 0
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 卡尔玛比率
        calmar = abs(np.mean(returns) * 365 / max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'volatility': np.std(returns) * np.sqrt(365),
            'calmar_ratio': calmar
        }
    
    def fixed_compound_calculation(self, trade_profit):
        """修复的复利计算"""
        if self.use_compound:
            # 使用当前资金的固定比例进行交易
            trade_amount = self.current_capital * self.position_size
            actual_profit = trade_profit * trade_amount / self.initial_capital
            self.current_capital += actual_profit
        else:
            # 不复利模式
            trade_amount = self.initial_capital * self.position_size
            actual_profit = trade_profit * trade_amount / self.initial_capital
            self.current_capital = self.initial_capital + sum(t['profit'] for t in self.trades) + actual_profit
        
        return actual_profit
    
    def run_backtest(self, data):
        """运行回测"""
        self.logger.info("🚀 启动修复版回测系统")
        self.logger.info(f"初始资金: ${self.initial_capital:,}")
        self.logger.info(f"仓位比例: {self.position_size*100}%")
        self.logger.info(f"复利模式: {self.use_compound}")
        
        # 模拟一些交易数据用于测试
        # 这里应该替换为实际的交易逻辑
        sample_trades = [
            {'profit': 0.02, 'duration': 1},   # 2% 收益
            {'profit': -0.01, 'duration': 1},  # -1% 亏损
            {'profit': 0.03, 'duration': 2},   # 3% 收益
            {'profit': 0.015, 'duration': 1},  # 1.5% 收益
            {'profit': -0.02, 'duration': 3},  # -2% 亏损
        ]
        
        for i, trade in enumerate(sample_trades):
            actual_profit = self.fixed_compound_calculation(trade['profit'])
            
            self.trades.append({
                'trade_id': i + 1,
                'profit_pct': trade['profit'] * 100,
                'profit_actual': actual_profit,
                'capital_after': self.current_capital,
                'duration': trade['duration']
            })
            
            self.portfolio_values.append(self.current_capital)
        
        # 计算收益序列用于风险评估
        returns = [trade['profit_pct'] / 100 for trade in self.trades]
        
        # 计算风险指标
        risk_metrics = self.calculate_risk_metrics(pd.Series(returns))
        
        return self.generate_report(risk_metrics)
    
    def generate_report(self, risk_metrics):
        """生成报告"""
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['profit_actual'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_profit = sum(t['profit_actual'] for t in self.trades)
        
        report = f"""
🔧 修复版复利回测报告
==================================================
📊 交易表现:
   总交易次数: {total_trades}笔
   盈利交易: {winning_trades}笔
   胜率: {win_rate:.1%}
   总收益: ${total_profit:+.2f}
   最终资金: ${self.current_capital:,.2f}
   收益率: {(self.current_capital - self.initial_capital) / self.initial_capital:.1%}

⚡ 风险评估:
   夏普比率: {risk_metrics['sharpe_ratio']:.2f}
   最大回撤: {risk_metrics['max_drawdown']:.1%}
   年化波动率: {risk_metrics['volatility']:.1%}
   卡尔玛比率: {risk_metrics['calmar_ratio']:.2f}

🎯 参数设置:
   初始资金: ${self.initial_capital:,}
   仓位比例: {self.position_size*100}%
   复利模式: {self.use_compound}
"""
        self.logger.info(report)
        return report

# 使用示例
if __name__ == "__main__":
    # 测试修复版本
    backtester = FixedCompoundBacktest(
        initial_capital=10000,
        position_size=0.1,  # 10%仓位
        use_compound=True
    )
    
    # 运行回测 (这里需要实际的数据)
    sample_data = pd.DataFrame()  # 替换为实际数据
    result = backtester.run_backtest(sample_data)