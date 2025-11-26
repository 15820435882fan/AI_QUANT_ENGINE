# strategy_analyzer.py
#!/usr/bin/env python3
import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig, DataManager
from src.backtesting.backtest_strategies import RobustSMAStrategy
from src.strategies.macd_strategy import MACDStrategy

class StrategyAnalyzer:
    """策略详细分析器"""
    
    def __init__(self):
        self.results = {}
        
    async def analyze_strategy_detailed(self, strategy_name, strategy_class, params):
        """详细分析策略表现"""
        print(f"\n🔍 详细分析: {strategy_name}")
        print("=" * 50)
        
        config = BacktestConfig(
            initial_capital=10000.0,
            start_date="2024-01-01",
            end_date="2024-01-10"
        )
        
        engine = BacktestEngine(config)
        data_manager = DataManager()
        
        historical_data = await data_manager.load_historical_data(
            "BTC/USDT", config.start_date, config.end_date
        )
        
        # 创建策略实例
        strategy = strategy_class(
            name=strategy_name,
            symbols=["BTC/USDT"],
            **params
        )
        
        # 运行回测
        result = await engine.run_backtest(strategy, historical_data)
        
        # 详细分析
        analysis = self._detailed_analysis(result, strategy_name, historical_data)
        
        return analysis
    
    def _detailed_analysis(self, result, strategy_name, historical_data):
        """生成详细分析报告"""
        analysis = {
            'strategy_name': strategy_name,
            'total_return': result.total_return,
            'total_trades': result.total_trades,
            'max_drawdown': result.max_drawdown,
            'sharpe_ratio': result.sharpe_ratio,
            'win_rate': result.win_rate,
            'trades': result.trades,
            'equity_curve': result.equity_curve
        }
        
        # 交易统计
        if result.trades:
            buy_trades = [t for t in result.trades if t.get('signal_type') == 'buy']
            sell_trades = [t for t in result.trades if t.get('signal_type') == 'sell']
            
            analysis['buy_trades'] = len(buy_trades)
            analysis['sell_trades'] = len(sell_trades)
            analysis['avg_trade_duration'] = self._calculate_avg_duration(result.trades)
            
            # 盈利交易分析
            profitable_trades = [t for t in result.trades if t.get('profit', 0) > 0]
            analysis['profitable_trades'] = len(profitable_trades)
            analysis['profitability_rate'] = len(profitable_trades) / len(result.trades) if result.trades else 0
        
        return analysis
    
    def _calculate_avg_duration(self, trades):
        """计算平均持仓时间"""
        if len(trades) < 2:
            return 0
        
        durations = []
        for i in range(1, len(trades), 2):
            if i < len(trades):
                buy_time = trades[i-1]['timestamp']
                sell_time = trades[i]['timestamp']
                if hasattr(buy_time, 'timestamp') and hasattr(sell_time, 'timestamp'):
                    duration = (sell_time - buy_time).total_seconds() / 3600  # 小时
                    durations.append(duration)
        
        return np.mean(durations) if durations else 0
    
    def print_detailed_report(self, analysis):
        """打印详细报告"""
        print(f"\n📊 策略: {analysis['strategy_name']}")
        print(f"💰 总收益: {analysis['total_return']:.2%}")
        print(f"🔢 总交易数: {analysis['total_trades']}")
        print(f"📉 最大回撤: {analysis['max_drawdown']:.2%}")
        print(f"⚡ 夏普比率: {analysis['sharpe_ratio']:.2f}")
        print(f"🎯 胜率: {analysis['win_rate']:.2%}")
        
        if 'buy_trades' in analysis:
            print(f"🛒 买入交易: {analysis['buy_trades']}")
            print(f"🏪 卖出交易: {analysis['sell_trades']}")
            print(f"⏱️ 平均持仓: {analysis['avg_trade_duration']:.1f}小时")
            print(f"💹 盈利交易: {analysis['profitable_trades']} ({analysis['profitability_rate']:.1%})")
        
        # 显示前5笔交易
        if analysis['trades']:
            print(f"\n📋 前5笔交易:")
            for i, trade in enumerate(analysis['trades'][:5]):
                status = trade.get('status', 'executed')
                print(f"  {i+1}. {trade['timestamp']} {trade['signal_type']} {trade['quantity']:.4f} @ {trade['price']:.2f} - {status}")

async def adaptive_strategy_test():
    """测试自适应策略选择"""
    print("🎯 测试自适应策略选择系统")
    print("=" * 60)
    
    from src.strategies.multi_strategy_manager import MultiStrategyManager
    from src.strategies.market_regime_detector import MarketRegimeDetector
    
    # 创建组件
    strategy_manager = MultiStrategyManager()
    regime_detector = MarketRegimeDetector()
    
    # 加载数据
    data_manager = DataManager()
    historical_data = await data_manager.load_historical_data(
        "BTC/USDT", "2024-01-01", "2024-01-05"
    )
    
    print(f"📊 加载 {len(historical_data)} 条历史数据")
    
    # 分析市场状态变化
    regime_changes = []
    window_size = 1440  # 24小时数据
    
    for i in range(window_size, len(historical_data), 360):  # 每6小时检测一次
        window_data = historical_data.iloc[i-window_size:i]
        
        try:
            regime = await regime_detector.detect_regime(window_data)
            regime_changes.append({
                'timestamp': historical_data.index[i],
                'regime': regime,
                'price': historical_data.iloc[i]['close']
            })
        except Exception as e:
            print(f"市场状态检测错误: {e}")
            continue
    
    # 分析市场状态分布
    if regime_changes:
        regimes = [r['regime'] for r in regime_changes]
        print(f"\n🌍 市场状态分析:")
        for regime in set(regimes):
            count = regimes.count(regime)
            percentage = count / len(regimes) * 100
            print(f"  {regime}: {count}次 ({percentage:.1f}%)")
    
    return regime_changes

async def main():
    """主分析函数"""
    print("🧠 开始策略详细分析")
    
    analyzer = StrategyAnalyzer()
    
    # 定义要测试的策略
    strategies = [
        ("SMA策略", RobustSMAStrategy, {"fast_period": 10, "slow_period": 30}),
        ("MACD标准", MACDStrategy, {"fast_period": 12, "slow_period": 26, "signal_period": 9}),
        ("MACD快速", MACDStrategy, {"fast_period": 6, "slow_period": 19, "signal_period": 5}),
    ]
    
    # 分析每个策略
    all_analysis = {}
    for strategy_name, strategy_class, params in strategies:
        analysis = await analyzer.analyze_strategy_detailed(strategy_name, strategy_class, params)
        all_analysis[strategy_name] = analysis
        analyzer.print_detailed_report(analysis)
    
    # 找出最佳策略
    best_strategy = max(all_analysis.items(), key=lambda x: x[1]['total_return'])
    
    print(f"\n🎉 综合评估:")
    print(f"🏆 最佳策略: {best_strategy[0]} ({best_strategy[1]['total_return']:.2%})")
    
    # 测试自适应系统
    print(f"\n" + "=" * 60)
    await adaptive_strategy_test()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())