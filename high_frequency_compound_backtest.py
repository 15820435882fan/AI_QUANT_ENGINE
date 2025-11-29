# high_frequency_compound_backtest.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import sys
import os

class HighFrequencyCompoundBacktest:
    def __init__(self, initial_capital=10000, base_position_size=0.1, 
                 target_daily_return=0.01, max_daily_trades=10):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.base_position_size = base_position_size
        self.target_daily_return = target_daily_return  # 日均1%目标
        self.max_daily_trades = max_daily_trades
        
        self.trades = []
        self.daily_balances = [initial_capital]
        self.daily_summary = []
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def adaptive_position_sizing(self, signal_quality, current_daily_return):
        """自适应仓位调整"""
        base_size = self.base_position_size
        
        # 根据信号质量调整
        if signal_quality > 1.2:
            size_multiplier = 1.5  # 强信号
        elif signal_quality > 0.8:
            size_multiplier = 1.0  # 中等信号
        else:
            size_multiplier = 0.5  # 弱信号
            
        # 根据当日收益目标调整
        if current_daily_return < self.target_daily_return * 0.5:
            # 离目标较远，适度增加风险
            target_multiplier = 1.2
        elif current_daily_return > self.target_daily_return * 1.5:
            # 已超目标，降低风险
            target_multiplier = 0.7
        else:
            target_multiplier = 1.0
            
        final_size = base_size * size_multiplier * target_multiplier
        return min(final_size, 0.3)  # 最大30%仓位

    def generate_realistic_trades(self, days=30, trades_per_day=8):
        """生成更真实的高频交易数据"""
        all_trades = []
        
        for day in range(days):
            daily_trades = []
            daily_profit = 0
            
            for trade_num in range(trades_per_day):
                # 模拟更真实的收益分布
                if np.random.random() > 0.4:  # 60%胜率
                    # 盈利交易: 0.1% - 1.5%
                    profit_pct = np.random.uniform(0.001, 0.015)
                    signal_quality = np.random.uniform(1.0, 1.5)
                else:
                    # 亏损交易: -0.05% - -0.8%
                    profit_pct = np.random.uniform(-0.008, -0.0005)
                    signal_quality = np.random.uniform(0.5, 0.9)
                
                daily_trades.append({
                    'profit_pct': profit_pct,
                    'signal_quality': signal_quality,
                    'day': day,
                    'trade_num': trade_num
                })
                daily_profit += profit_pct
            
            all_trades.extend(daily_trades)
            
        return all_trades

    def execute_high_frequency_strategy(self, days=30):
        """执行高频交易策略"""
        self.logger.info(f"🚀 启动高频复利回测 - {days}天")
        self.logger.info(f"目标: 日均{self.target_daily_return:.1%} | 最大{self.max_daily_trades}笔/天")
        
        # 生成交易数据
        trade_plan = self.generate_realistic_trades(days, self.max_daily_trades)
        
        current_day = 0
        daily_trade_count = 0
        daily_return = 0
        
        for i, trade_info in enumerate(trade_plan):
            # 新一天重置
            if trade_info['day'] != current_day:
                current_day = trade_info['day']
                daily_trade_count = 0
                daily_return = 0
                self.daily_balances.append(self.current_capital)
            
            # 自适应仓位
            position_size = self.adaptive_position_sizing(
                trade_info['signal_quality'], 
                daily_return
            )
            
            # 执行交易
            trade_amount = self.current_capital * position_size
            actual_profit = trade_amount * trade_info['profit_pct']
            self.current_capital += actual_profit
            
            # 更新日度统计
            daily_return = (self.current_capital - self.daily_balances[-1]) / self.daily_balances[-1]
            daily_trade_count += 1
            
            # 记录交易
            trade_record = {
                'trade_id': i + 1,
                'day': current_day,
                'profit_pct': trade_info['profit_pct'] * 100,
                'profit_actual': actual_profit,
                'position_size': position_size,
                'signal_quality': trade_info['signal_quality'],
                'capital_after': self.current_capital,
                'daily_return': daily_return
            }
            self.trades.append(trade_record)
            
            # 每日交易限制
            if daily_trade_count >= self.max_daily_trades:
                continue
        
        return self.generate_high_frequency_report(days)

    def calculate_high_frequency_metrics(self, days):
        """计算高频交易专属指标"""
        if len(self.trades) == 0:
            return self._empty_metrics()
        
        # 基础指标
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        daily_returns = self._calculate_daily_returns()
        
        # 高频专属指标
        winning_days = len([dr for dr in daily_returns if dr > 0])
        daily_win_rate = winning_days / len(daily_returns) if len(daily_returns) > 0 else 0
        
        avg_daily_trades = len(self.trades) / days
        avg_daily_return = np.mean(daily_returns) if len(daily_returns) > 0 else 0
        
        # 连续盈利/亏损分析
        consecutive_stats = self._calculate_consecutive_stats(daily_returns)
        
        return {
            'total_trades': len(self.trades),
            'total_days': days,
            'avg_daily_trades': avg_daily_trades,
            'total_return': total_return,
            'annualized_return': total_return / days * 365,
            'avg_daily_return': avg_daily_return,
            'daily_win_rate': daily_win_rate,
            'consecutive_winning_days': consecutive_stats['max_win_streak'],
            'consecutive_losing_days': consecutive_stats['max_loss_streak'],
            **self._calculate_risk_metrics(daily_returns)
        }

    def _calculate_consecutive_stats(self, daily_returns):
        """计算连续盈利/亏损统计"""
        if len(daily_returns) == 0:
            return {'max_win_streak': 0, 'max_loss_streak': 0}
        
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        current_type = None
        
        for ret in daily_returns:
            if ret > 0:  # 盈利日
                if current_type == 'win':
                    current_streak += 1
                else:
                    current_streak = 1
                    current_type = 'win'
                max_win_streak = max(max_win_streak, current_streak)
            else:  # 亏损日
                if current_type == 'loss':
                    current_streak += 1
                else:
                    current_streak = 1
                    current_type = 'loss'
                max_loss_streak = max(max_loss_streak, current_streak)
        
        return {
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak
        }

    def generate_high_frequency_report(self, days):
        """生成高频交易报告"""
        metrics = self.calculate_high_frequency_metrics(days)
        
        report = f"""
🎯 高频复利交易报告 ({days}天)
==================================================
📊 交易统计:
   总交易次数: {metrics['total_trades']:,}笔
   交易天数: {metrics['total_days']}天
   日均交易: {metrics['avg_daily_trades']:.1f}笔
   日均胜率: {metrics['daily_win_rate']:.1%}

💰 收益表现:
   总收益率: {metrics['total_return']:.1%}
   年化收益率: {metrics['annualized_return']:.1%}
   日均收益率: {metrics['avg_daily_return']:.3%}
   最终资金: ${self.current_capital:,.2f}

⚡ 风险指标:
   夏普比率: {metrics['sharpe_ratio']:.2f}
   最大回撤: {metrics['max_drawdown']:.1%}
   最长盈利天数: {metrics['consecutive_winning_days']}天
   最长亏损天数: {metrics['consecutive_losing_days']}天

🎯 目标评估:
   日均1%目标: {'✅ 达成' if metrics['avg_daily_return'] >= 0.01 else '❌ 未达'}
   资金增长: {self.current_capital/self.initial_capital:.1f}倍
   
📈 策略评级: {'🔥 优秀' if metrics['annualized_return'] > 1.0 else '✅ 良好' if metrics['annualized_return'] > 0.5 else '⚠️ 需优化'}
"""
        self.logger.info(report)
        
        # 打印最近10笔交易示例
        self.logger.info("\n📋 最近10笔交易示例:")
        for trade in self.trades[-10:]:
            self.logger.info(f"   交易{trade['trade_id']}: {trade['profit_pct']:+.3f}% | 资金: ${trade['capital_after']:,.2f}")
        
        return report

    def _calculate_daily_returns(self):
        """计算日度收益率"""
        if len(self.daily_balances) < 2:
            return pd.Series([0])
        balances = pd.Series(self.daily_balances)
        return balances.pct_change().dropna()

    def _calculate_risk_metrics(self, daily_returns):
        """计算风险指标"""
        if len(daily_returns) == 0:
            return {'sharpe_ratio': 0, 'max_drawdown': 0}
        
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365) if np.std(daily_returns) > 0 else 0
        
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown
        }

    def _empty_metrics(self):
        return {
            'total_trades': 0, 'total_days': 0, 'avg_daily_trades': 0,
            'total_return': 0, 'annualized_return': 0, 'avg_daily_return': 0,
            'daily_win_rate': 0, 'consecutive_winning_days': 0, 'consecutive_losing_days': 0,
            'sharpe_ratio': 0, 'max_drawdown': 0
        }

def main():
    """测试高频版本"""
    # 测试30天高频交易
    hf_backtest = HighFrequencyCompoundBacktest(
        initial_capital=10000,
        base_position_size=0.15,  # 15%基础仓位
        target_daily_return=0.01,  # 日均1%目标
        max_daily_trades=8        # 每天最多8笔交易
    )
    
    hf_backtest.execute_high_frequency_strategy(days=30)

if __name__ == "__main__":
    main()