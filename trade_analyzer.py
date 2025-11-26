# trade_analyzer.py
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import logging

class TradeAnalyzer:
    """交易分析器 - 深入分析每笔交易"""
    
    def __init__(self):
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('TradeAnalyzer')
    
    def analyze_trade_details(self, trade_history: List[Dict]) -> pd.DataFrame:
        """分析交易细节"""
        if not trade_history:
            return pd.DataFrame()
        
        trades_df = pd.DataFrame(trade_history)
        
        # 计算详细指标
        analysis_results = []
        for i, trade in enumerate(trade_history):
            if trade['action'] == 'CLOSE':
                analysis = {
                    '序号': i + 1,
                    '币种': trade.get('symbol', 'N/A'),
                    '方向': trade.get('direction', 'N/A'),
                    '开仓时间': trade.get('entry_time', 'N/A'),
                    '开仓价格': trade.get('entry_price', 0),
                    '平仓时间': trade.get('exit_time', 'N/A'),
                    '平仓价格': trade.get('exit_price', 0),
                    '持仓时间': self._calculate_holding_period(trade),
                    '仓位大小': f"${trade.get('position_size', 0):.0f}",
                    '杠杆': f"{trade.get('leverage', 0)}x",
                    '盈亏': f"${trade.get('pnl', 0):+.0f}",
                    '盈亏百分比': f"{trade.get('pnl_pct', 0):+.1f}%",
                    '平仓原因': trade.get('reason', 'N/A'),
                    '置信度': f"{trade.get('confidence', 0):.1%}"
                }
                analysis_results.append(analysis)
        
        return pd.DataFrame(analysis_results)
    
    def _calculate_holding_period(self, trade: Dict) -> str:
        """计算持仓时间"""
        entry_time = trade.get('entry_time')
        exit_time = trade.get('exit_time')
        
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
        if isinstance(exit_time, str):
            exit_time = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
        
        if entry_time and exit_time:
            duration = exit_time - entry_time
            hours = duration.total_seconds() / 3600
            if hours < 1:
                return f"{duration.total_seconds()/60:.0f}分钟"
            elif hours < 24:
                return f"{hours:.1f}小时"
            else:
                return f"{hours/24:.1f}天"
        return "N/A"
    
    def generate_trade_report(self, trade_history: List[Dict], symbol: str, period: str):
        """生成交易报告"""
        print(f"\n{'='*80}")
        print(f"📋 详细交易报告 - {symbol} - {period}")
        print(f"{'='*80}")
        
        trades_df = self.analyze_trade_details(trade_history)
        
        if trades_df.empty:
            print("暂无交易记录")
            return
        
        # 显示所有交易
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(trades_df.to_string(index=False))
        
        # 统计分析
        self._print_trade_statistics(trades_df, symbol, period)
    
    def _print_trade_statistics(self, trades_df: pd.DataFrame, symbol: str, period: str):
        """打印交易统计"""
        if trades_df.empty:
            return
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['盈亏'].str.contains('\+')])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades
        
        # 提取盈亏数值
        pnl_values = []
        for pnl_str in trades_df['盈亏']:
            try:
                pnl = float(pnl_str.replace('$', '').replace('+', ''))
                pnl_values.append(pnl)
            except:
                continue
        
        total_pnl = sum(pnl_values)
        avg_pnl = np.mean(pnl_values) if pnl_values else 0
        
        print(f"\n📊 {symbol} - {period} 交易统计:")
        print(f"  总交易次数: {total_trades}")
        print(f"  盈利交易: {winning_trades}")
        print(f"  亏损交易: {losing_trades}")
        print(f"  🎯 胜率: {win_rate:.1%}")
        print(f"  💰 总盈亏: ${total_pnl:+.0f}")
        print(f"  📊 平均每笔盈亏: ${avg_pnl:+.0f}")
        
        if pnl_values:
            print(f"  📈 最大盈利: ${max(pnl_values):.0f}")
            print(f"  📉 最大亏损: ${min(pnl_values):.0f}")