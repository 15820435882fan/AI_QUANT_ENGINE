# performance_dashboard.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

class PerformanceDashboard:
    """性能分析仪表板"""
    
    def __init__(self):
        self.setup_plotting()
    
    def setup_plotting(self):
        """设置绘图样式"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_trading_report(self, trade_history: List[Dict], 
                              portfolio_values: List[Dict]) -> Dict[str, Any]:
        """生成交易报告"""
        report = {}
        
        if not trade_history or not portfolio_values:
            return report
        
        # 基础统计
        report['total_trades'] = len(trade_history)
        report['buy_trades'] = len([t for t in trade_history if t['action'] == 'BUY'])
        report['sell_trades'] = len([t for t in trade_history if t['action'] == 'SELL'])
        
        # 计算胜率
        profitable_trades = [t for t in trade_history 
                           if t.get('profit_loss', 0) > 0]
        report['win_rate'] = len(profitable_trades) / report['sell_trades'] if report['sell_trades'] > 0 else 0
        
        # 计算平均盈亏
        if profitable_trades:
            report['avg_profit'] = np.mean([t['profit_loss'] for t in profitable_trades])
        
        losing_trades = [t for t in trade_history 
                        if t.get('profit_loss', 0) < 0]
        if losing_trades:
            report['avg_loss'] = np.mean([t['profit_loss'] for t in losing_trades])
        
        # 计算夏普比率
        returns = self.calculate_returns(portfolio_values)
        if len(returns) > 1:
            report['sharpe_ratio'] = returns.mean() / returns.std() if returns.std() > 0 else 0
            report['total_return'] = (portfolio_values[-1]['total_value'] - portfolio_values[0]['total_value']) / portfolio_values[0]['total_value']
        
        return report
    
    def calculate_returns(self, portfolio_values: List[Dict]) -> pd.Series:
        """计算收益率序列"""
        values = [pv['total_value'] for pv in portfolio_values]
        returns = pd.Series(values).pct_change().dropna()
        return returns
    
    def plot_portfolio_performance(self, portfolio_values: List[Dict], save_path: str = None):
        """绘制投资组合性能图表"""
        if len(portfolio_values) < 2:
            print("⚠️ 数据不足，无法生成图表")
            return
        
        dates = [pv.get('day', i) for i, pv in enumerate(portfolio_values)]
        values = [pv['total_value'] for pv in portfolio_values]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 投资组合价值曲线
        ax1.plot(dates, values, linewidth=2, label='投资组合价值')
        ax1.set_title('投资组合价值变化', fontsize=14, fontweight='bold')
        ax1.set_ylabel('价值 ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 收益率分布
        returns = self.calculate_returns(portfolio_values)
        ax2.hist(returns, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_title('收益率分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel('日收益率')
        ax2.set_ylabel('频率')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 图表已保存: {save_path}")
        
        plt.show()
    
    def plot_trade_analysis(self, trade_history: List[Dict], save_path: str = None):
        """绘制交易分析图表"""
        if not trade_history:
            print("⚠️ 无交易数据")
            return
        
        # 筛选卖出交易（有盈亏数据）
        sell_trades = [t for t in trade_history if t['action'] == 'SELL']
        
        if not sell_trades:
            print("⚠️ 无卖出交易数据")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 盈亏分布
        profits = [t.get('profit_loss', 0) for t in sell_trades]
        ax1.hist(profits, bins=15, alpha=0.7, edgecolor='black')
        ax1.set_title('交易盈亏分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel('盈亏金额 ($)')
        ax1.set_ylabel('交易次数')
        ax1.axvline(0, color='red', linestyle='--', alpha=0.8)
        ax1.grid(True, alpha=0.3)
        
        # 置信度 vs 盈亏
        confidences = [t.get('confidence', 0) for t in sell_trades]
        ax2.scatter(confidences, profits, alpha=0.6)
        ax2.set_title('置信度 vs 盈亏', fontsize=14, fontweight='bold')
        ax2.set_xlabel('交易置信度')
        ax2.set_ylabel('盈亏金额 ($)')
        ax2.axhline(0, color='red', linestyle='--', alpha=0.8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 交易分析图已保存: {save_path}")
        
        plt.show()
    
    def generate_comprehensive_report(self, paper_trader, save_dir: str = "reports"):
        """生成综合报告"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("📋 生成综合性能报告...")
        
        # 生成交易报告
        trade_report = self.generate_trading_report(
            paper_trader.trade_history, 
            paper_trader.portfolio_value
        )
        
        # 绘制图表
        portfolio_chart_path = f"{save_dir}/portfolio_performance.png"
        trade_analysis_path = f"{save_dir}/trade_analysis.png"
        
        self.plot_portfolio_performance(paper_trader.portfolio_value, portfolio_chart_path)
        self.plot_trade_analysis(paper_trader.trade_history, trade_analysis_path)
        
        # 输出报告
        print(f"\n{'='*50}")
        print(f"🎯 交易性能综合报告")
        print(f"{'='*50}")
        for key, value in trade_report.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print(f"\n📊 图表文件:")
        print(f"  投资组合性能: {portfolio_chart_path}")
        print(f"  交易分析: {trade_analysis_path}")

def test_dashboard():
    """测试仪表板"""
    print("🧪 测试性能分析仪表板...")
    
    from paper_trading_system import test_paper_trading
    
    # 运行模拟交易获取数据
    paper_trader = test_paper_trading()
    
    # 生成报告
    dashboard = PerformanceDashboard()
    dashboard.generate_comprehensive_report(paper_trader)
    
    return dashboard

if __name__ == "__main__":
    test_dashboard()