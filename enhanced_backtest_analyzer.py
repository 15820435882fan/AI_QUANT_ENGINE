# enhanced_backtest_analyzer.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import matplotlib.pyplot as plt

class EnhancedBacktestAnalyzer:
    """增强版回测分析器 - 详细统计分析"""
    
    def __init__(self):
        self.results = {}
        self.monthly_stats = {}
    
    def analyze_trade_results(self, trade_history: List, symbol: str, period: str):
        """分析交易结果"""
        if not trade_history:
            return
        
        # 按月份分组
        trades_df = pd.DataFrame(trade_history)
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        trades_df['month'] = trades_df['exit_time'].dt.to_period('M')
        
        # 月度统计
        monthly_analysis = {}
        for month, month_trades in trades_df.groupby('month'):
            monthly_stats = self._calculate_monthly_stats(month_trades, symbol, str(month))
            monthly_analysis[str(month)] = monthly_stats
        
        # 总体统计
        overall_stats = self._calculate_overall_stats(trades_df, symbol, period)
        
        self.results[symbol] = {
            'period': period,
            'overall': overall_stats,
            'monthly': monthly_analysis,
            'trades': trade_history
        }
    
    def _calculate_monthly_stats(self, trades_df: pd.DataFrame, symbol: str, month: str) -> Dict[str, Any]:
        """计算月度统计"""
        if trades_df.empty:
            return {
                'symbol': symbol,
                'month': month,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'max_win': 0,
                'max_loss': 0
            }
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        return {
            'symbol': symbol,
            'month': month,
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades_df),
            'total_pnl': trades_df['pnl'].sum(),
            'avg_pnl': trades_df['pnl'].mean(),
            'avg_win': winning_trades['pnl'].mean() if not winning_trades.empty else 0,
            'avg_loss': losing_trades['pnl'].mean() if not losing_trades.empty else 0,
            'max_win': winning_trades['pnl'].max() if not winning_trades.empty else 0,
            'max_loss': losing_trades['pnl'].min() if not losing_trades.empty else 0,
            'avg_holding_hours': trades_df['holding_hours'].mean()
        }
    
    def _calculate_overall_stats(self, trades_df: pd.DataFrame, symbol: str, period: str) -> Dict[str, Any]:
        """计算总体统计"""
        if trades_df.empty:
            return {
                'symbol': symbol,
                'period': period,
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0
            }
        
        # 基础统计
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        total_pnl = trades_df['pnl'].sum()
        win_rate = len(winning_trades) / len(trades_df)
        
        # 夏普比率（简化）
        sharpe_ratio = total_pnl / (trades_df['pnl'].std() * np.sqrt(len(trades_df))) if len(trades_df) > 1 else 0
        
        # 最大回撤（简化）
        cumulative_pnl = trades_df['pnl'].cumsum()
        max_drawdown = (cumulative_pnl.cummax() - cumulative_pnl).max()
        
        # 盈利因子
        profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if not losing_trades.empty else float('inf')
        
        return {
            'symbol': symbol,
            'period': period,
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'avg_trade_pnl': trades_df['pnl'].mean(),
            'avg_winning_trade': winning_trades['pnl'].mean() if not winning_trades.empty else 0,
            'avg_losing_trade': losing_trades['pnl'].mean() if not losing_trades.empty else 0
        }
    
    def generate_detailed_report(self):
        """生成详细报告"""
        print(f"\n{'='*120}")
        print(f"🎯 高频交易系统 - 详细回测报告")
        print(f"{'='*120}")
        
        # 按币种显示
        for symbol, result in self.results.items():
            overall = result['overall']
            monthly = result['monthly']
            
            print(f"\n📊 币种: {symbol} - {result['period']}")
            print(f"{'-'*80}")
            print(f"总交易: {overall['total_trades']:3d} | "
                  f"胜率: {overall['win_rate']:6.1%} | "
                  f"总收益: ${overall['total_pnl']:8.0f} | "
                  f"夏普: {overall['sharpe_ratio']:5.2f} | "
                  f"回撤: ${overall['max_drawdown']:6.0f}")
            
            # 月度明细
            if monthly:
                print(f"\n📅 月度明细:")
                for month, stats in monthly.items():
                    if stats['total_trades'] > 0:
                        print(f"  {month}: "
                              f"交易{stats['total_trades']:2d} | "
                              f"胜率{stats['win_rate']:5.1%} | "
                              f"收益${stats['total_pnl']:6.0f} | "
                              f"均盈${stats['avg_win']:5.0f} | "
                              f"均亏${stats['avg_loss']:5.0f}")
        
        # 汇总统计
        self._generate_summary_statistics()
    
    def _generate_summary_statistics(self):
        """生成汇总统计"""
        print(f"\n{'='*80}")
        print(f"📈 系统汇总统计")
        print(f"{'='*80}")
        
        total_trades = 0
        total_pnl = 0
        winning_months = 0
        total_months = 0
        
        for symbol, result in self.results.items():
            overall = result['overall']
            monthly = result['monthly']
            
            total_trades += overall['total_trades']
            total_pnl += overall['total_pnl']
            total_months += len(monthly)
            winning_months += sum(1 for m in monthly.values() if m['total_pnl'] > 0)
        
        if total_trades > 0:
            avg_win_rate = sum(r['overall']['win_rate'] for r in self.results.values()) / len(self.results)
            monthly_win_rate = winning_months / total_months if total_months > 0 else 0
            
            print(f"总交易次数: {total_trades}")
            print(f"平均胜率: {avg_win_rate:.1%}")
            print(f"总收益: ${total_pnl:,.0f}")
            print(f"盈利月份比例: {monthly_win_rate:.1%}")
            print(f"覆盖币种数: {len(self.results)}")