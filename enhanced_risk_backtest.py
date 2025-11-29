# enhanced_risk_backtest.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import sys
import os

# 添加现有模块路径
sys.path.append(os.path.dirname(__file__))

class EnhancedRiskBacktest:
    def __init__(self, initial_capital=10000, position_size=0.1, use_compound=True):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.use_compound = use_compound
        self.current_capital = initial_capital
        self.trades = []
        self.daily_balances = [initial_capital]
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def calculate_comprehensive_metrics(self):
        """计算全面的风险指标"""
        if len(self.trades) == 0:
            return self._empty_metrics()
        
        # 基础统计
        profits = [t['profit_actual'] for t in self.trades]
        profits_pct = [t['profit_pct'] for t in self.trades]
        
        total_profit = sum(profits)
        winning_trades = len([p for p in profits if p > 0])
        win_rate = winning_trades / len(profits)
        
        # 日度收益计算 (简化版)
        daily_returns = self._calculate_daily_returns()
        
        # 风险指标
        risk_metrics = self._calculate_risk_metrics(daily_returns)
        
        # 交易质量指标
        trade_metrics = self._calculate_trade_metrics(profits, profits_pct)
        
        return {
            **risk_metrics,
            **trade_metrics,
            'total_trades': len(self.trades),
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'final_capital': self.current_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital
        }

    def _calculate_daily_returns(self):
        """计算日度收益率"""
        if len(self.daily_balances) < 2:
            return pd.Series([0])
        
        balances = pd.Series(self.daily_balances)
        daily_returns = balances.pct_change().dropna()
        return daily_returns

    def _calculate_risk_metrics(self, daily_returns):
        """计算风险指标"""
        if len(daily_returns) == 0:
            return {
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'volatility_annual': 0,
                'calmar_ratio': 0,
                'sortino_ratio': 0
            }
        
        # 夏普比率
        excess_returns = daily_returns - 0  # 无风险利率为0
        sharpe = np.mean(excess_returns) / np.std(daily_returns) * np.sqrt(365) if np.std(daily_returns) > 0 else 0
        
        # 索提诺比率 (只考虑下行风险)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino = np.mean(excess_returns) / downside_std * np.sqrt(365) if downside_std > 0 else 0
        
        # 最大回撤
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 卡尔玛比率
        annual_return = np.mean(daily_returns) * 365
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'volatility_annual': np.std(daily_returns) * np.sqrt(365),
            'calmar_ratio': calmar
        }

    def _calculate_trade_metrics(self, profits, profits_pct):
        """计算交易质量指标"""
        profits_pct_series = pd.Series(profits_pct)
        
        return {
            'avg_profit_per_trade': np.mean(profits),
            'profit_factor': abs(sum(p for p in profits if p > 0) / sum(p for p in profits if p < 0)) if any(p < 0 for p in profits) else float('inf'),
            'expectancy': (profits_pct_series.mean() * profits_pct_series[profits_pct_series > 0].count() / len(profits_pct_series)) - 
                         (abs(profits_pct_series[profits_pct_series < 0].mean()) * profits_pct_series[profits_pct_series < 0].count() / len(profits_pct_series)) if len(profits_pct_series) > 0 else 0,
            'avg_winning_trade': profits_pct_series[profits_pct_series > 0].mean() if any(p > 0 for p in profits_pct) else 0,
            'avg_losing_trade': profits_pct_series[profits_pct_series < 0].mean() if any(p < 0 for p in profits_pct) else 0
        }

    def _empty_metrics(self):
        """空交易时的默认指标"""
        return {
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'volatility_annual': 0,
            'calmar_ratio': 0,
            'avg_profit_per_trade': 0,
            'profit_factor': 0,
            'expectancy': 0,
            'avg_winning_trade': 0,
            'avg_losing_trade': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'win_rate': 0,
            'total_profit': 0,
            'final_capital': self.initial_capital,
            'total_return': 0
        }

    def fixed_position_sizing(self, signal_strength=1.0):
        """修复的仓位大小计算"""
        base_size = self.position_size
        adjusted_size = min(base_size * signal_strength, 0.3)  # 最大30%仓位
        return adjusted_size

    def execute_trade(self, profit_percentage, signal_strength=1.0):
        """执行交易并更新资金"""
        position_size = self.fixed_position_sizing(signal_strength)
        
        if self.use_compound:
            # 复利模式：使用当前资金计算
            trade_amount = self.current_capital * position_size
            actual_profit = trade_amount * profit_percentage
            self.current_capital += actual_profit
        else:
            # 非复利模式：使用初始资金计算
            trade_amount = self.initial_capital * position_size
            actual_profit = trade_amount * profit_percentage
            self.current_capital = self.initial_capital + sum(t['profit_actual'] for t in self.trades) + actual_profit
        
        # 记录交易
        trade = {
            'profit_pct': profit_percentage * 100,
            'profit_actual': actual_profit,
            'position_size': position_size,
            'capital_after': self.current_capital,
            'timestamp': datetime.now()
        }
        self.trades.append(trade)
        
        # 更新日度余额
        self.daily_balances.append(self.current_capital)
        
        return trade

    def run_test_scenario(self):
        """运行测试场景"""
        self.logger.info("🚀 启动增强版风险回测系统")
        
        # 模拟真实交易场景
        test_trades = [
            (0.025, 1.0),   # 2.5%收益，强信号
            (-0.015, 0.8),  # -1.5%亏损，中等信号  
            (0.035, 1.2),   # 3.5%收益，很强信号
            (0.018, 0.9),   # 1.8%收益，中等信号
            (-0.022, 0.7),  # -2.2%亏损，弱信号
            (0.028, 1.1),   # 2.8%收益，强信号
        ]
        
        for i, (profit_pct, signal_strength) in enumerate(test_trades):
            trade = self.execute_trade(profit_pct, signal_strength)
            self.logger.info(f"交易 {i+1}: {profit_pct:+.1%} | 资金: ${trade['capital_after']:,.2f}")
        
        return self.generate_comprehensive_report()

    def generate_comprehensive_report(self):
        """生成全面报告"""
        metrics = self.calculate_comprehensive_metrics()
        
        report = f"""
🎯 增强版风险回测报告
==================================================
📊 交易表现汇总:
   总交易次数: {metrics['total_trades']}笔
   盈利交易: {metrics['winning_trades']}笔
   胜率: {metrics['win_rate']:.1%}
   总收益: ${metrics['total_profit']:+.2f}
   最终资金: ${metrics['final_capital']:,.2f}
   总收益率: {metrics['total_return']:.1%}

⚡ 风险评估指标:
   夏普比率: {metrics['sharpe_ratio']:.2f} {'✅' if metrics['sharpe_ratio'] > 1.0 else '⚠️'}
   索提诺比率: {metrics['sortino_ratio']:.2f} {'✅' if metrics['sortino_ratio'] > 1.0 else '⚠️'}
   最大回撤: {metrics['max_drawdown']:.1%} {'✅' if metrics['max_drawdown'] > -0.15 else '⚠️'}
   年化波动率: {metrics['volatility_annual']:.1%}
   卡尔玛比率: {metrics['calmar_ratio']:.2f} {'✅' if metrics['calmar_ratio'] > 1.0 else '⚠️'}

💹 交易质量分析:
   平均每笔收益: ${metrics['avg_profit_per_trade']:+.2f}
   盈利因子: {metrics['profit_factor']:.2f} {'✅' if metrics['profit_factor'] > 1.5 else '⚠️'}
   期望值: {metrics['expectancy']:.2f}% {'✅' if metrics['expectancy'] > 0 else '❌'}
   平均盈利: {metrics['avg_winning_trade']:.2f}%
   平均亏损: {metrics['avg_losing_trade']:.2f}%

🎪 参数配置:
   初始资金: ${self.initial_capital:,}
   基础仓位: {self.position_size*100}%
   复利模式: {self.use_compound}
   
📈 绩效评级: {'优秀' if metrics['sharpe_ratio'] > 1.5 and metrics['win_rate'] > 0.5 else '良好' if metrics['sharpe_ratio'] > 1.0 else '需要优化'}
"""
        self.logger.info(report)
        return report

def main():
    """主函数 - 测试修复版本"""
    # 测试不同配置
    configs = [
        {"position_size": 0.1, "use_compound": True},
        {"position_size": 0.1, "use_compound": False},
        {"position_size": 0.2, "use_compound": True},
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*50}")
        print(f"测试配置 {i}: 仓位{config['position_size']*100}% | 复利{config['use_compound']}")
        print(f"{'='*50}")
        
        backtester = EnhancedRiskBacktest(
            initial_capital=10000,
            position_size=config['position_size'],
            use_compound=config['use_compound']
        )
        
        backtester.run_test_scenario()

if __name__ == "__main__":
    main()