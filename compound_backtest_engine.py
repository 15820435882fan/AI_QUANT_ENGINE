# compound_backtest_engine.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Tuple
from adaptive_compound_engine import AdaptiveCompoundEngine
from src.strategies.trend_following_compound import TrendFollowingCompound
from src.strategies.mean_reversion_compound import MeanReversionCompound

class CompoundBacktestEngine:
    """复利引擎回测系统"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.results = {}
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('CompoundBacktest')
    
    def generate_crypto_data_2024(self, symbol: str, periods: int = 365) -> pd.DataFrame:
        """生成2024年加密货币真实风格数据"""
        np.random.seed(hash(symbol) % 10000)
        
        # 2024年各币种基础价格和特征
        crypto_profiles = {
            'BTC-USDT': {'start_price': 45000, 'volatility': 0.025, 'trend': 0.0015},
            'ETH-USDT': {'start_price': 2500, 'volatility': 0.03, 'trend': 0.0012},
            'DOGE-USDT': {'start_price': 0.08, 'volatility': 0.05, 'trend': 0.0008}
        }
        
        profile = crypto_profiles.get(symbol, {'start_price': 100, 'volatility': 0.02, 'trend': 0.001})
        
        prices = [profile['start_price']]
        dates = []
        
        # 生成每日数据（2024年全年）
        start_date = datetime(2024, 1, 1)
        
        for day in range(periods):
            current_date = start_date + timedelta(days=day)
            dates.append(current_date)
            
            if day == 0:
                continue
                
            # 模拟真实市场特征 - 包含趋势、季节性和随机事件
            base_trend = profile['trend']
            
            # 季节性效应（季度末波动）
            seasonal = 0.002 * np.sin(2 * np.pi * day / 90)
            
            # 随机事件（5%概率出现大幅波动）
            event_impact = 0
            if np.random.random() < 0.05:
                event_impact = np.random.normal(0, 0.08)
            
            # 价格变化
            daily_change = np.random.normal(base_trend + seasonal, profile['volatility']) + event_impact
            new_price = prices[-1] * (1 + daily_change)
            
            # 防止价格归零，模拟真实支撑
            min_price = profile['start_price'] * 0.3
            prices.append(max(new_price, min_price))
        
        # 创建完整的OHLCV数据
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.015))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.015))) for p in prices],
            'close': prices,
            'volume': [np.random.randint(1000000, 50000000) for _ in prices]
        })
        
        data.set_index('timestamp', inplace=True)
        return data
    
    def run_backtest(self, symbol: str, periods: int = 365) -> Dict[str, Any]:
        """运行单个币种回测"""
        self.logger.info(f"🚀 开始回测 {symbol} - {periods}天")
        
        # 生成数据
        data = self.generate_crypto_data_2024(symbol, periods)
        
        # 创建复利引擎
        engine = AdaptiveCompoundEngine(initial_capital=self.initial_capital)
        
        # 添加策略
        trend_strategy = TrendFollowingCompound({
            'name': '趋势跟踪',
            'weight': 0.6,
            'parameters': {'fast_window': 10, 'slow_window': 30}
        })
        
        mean_reversion_strategy = MeanReversionCompound({
            'name': '均值回归', 
            'weight': 0.4,
            'parameters': {'bb_period': 20, 'bb_std': 2.0}
        })
        
        engine.add_strategy(trend_strategy)
        engine.add_strategy(mean_reversion_strategy)
        
        # 回测参数
        capital = self.initial_capital
        position = 0.0
        trade_history = []
        portfolio_values = []
        daily_returns = []
        
        # 按天回测
        unique_dates = data.index.normalize().unique()
        
        for i, current_date in enumerate(unique_dates):
            if i < 50:  # 前50天作为预热期
                continue
                
            # 获取到当前日期的所有数据
            current_data = data[data.index.normalize() <= current_date].tail(100)
            
            if len(current_data) < 50:
                continue
            
            current_price = current_data['close'].iloc[-1]
            
            # 生成交易信号
            signals = engine.generate_compound_signals(current_data)
            
            if 'error' in signals:
                continue
                
            decision = signals['decision']
            action = decision['action']
            position_size = decision['position_size']
            
            # 执行交易
            if action == 'BUY' and position == 0:
                # 开多仓
                trade_value = capital * position_size
                position = trade_value / current_price
                capital -= trade_value
                
                trade_history.append({
                    'date': current_date,
                    'action': 'BUY',
                    'price': current_price,
                    'quantity': position,
                    'value': trade_value,
                    'signal_confidence': decision['confidence']
                })
                
            elif action == 'SELL' and position > 0:
                # 平多仓
                trade_value = position * current_price
                profit_loss = trade_value - (position * trade_history[-1]['price'])
                profit_loss_pct = (profit_loss / (position * trade_history[-1]['price'])) * 100
                
                capital += trade_value
                
                trade_history.append({
                    'date': current_date,
                    'action': 'SELL',
                    'price': current_price,
                    'quantity': position,
                    'value': trade_value,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': profit_loss_pct,
                    'signal_confidence': decision['confidence']
                })
                
                position = 0
            
            # 计算投资组合价值
            portfolio_value = capital + (position * current_price)
            portfolio_values.append({
                'date': current_date,
                'value': portfolio_value,
                'cash': capital,
                'position': position
            })
            
            # 计算日收益率
            if len(portfolio_values) > 1:
                prev_value = portfolio_values[-2]['value']
                daily_return = (portfolio_value - prev_value) / prev_value
                daily_returns.append(daily_return)
        
        # 计算绩效指标
        metrics = self.calculate_performance_metrics(
            portfolio_values, trade_history, daily_returns
        )
        
        self.results[symbol] = {
            'metrics': metrics,
            'trade_history': trade_history,
            'portfolio_values': portfolio_values,
            'final_signals': signals
        }
        
        return self.results[symbol]
    
    def calculate_performance_metrics(self, portfolio_values: List, 
                                   trade_history: List, 
                                   daily_returns: List) -> Dict[str, float]:
        """计算绩效指标"""
        if not portfolio_values:
            return {}
            
        initial_value = self.initial_capital
        final_value = portfolio_values[-1]['value']
        total_return = (final_value - initial_value) / initial_value
        
        # 过滤出平仓交易
        closed_trades = [t for t in trade_history if t['action'] == 'SELL']
        
        # 胜率
        winning_trades = len([t for t in closed_trades if t.get('profit_loss', 0) > 0])
        win_rate = winning_trades / len(closed_trades) if closed_trades else 0
        
        # 平均盈亏比
        if closed_trades:
            avg_win = np.mean([t['profit_loss'] for t in closed_trades if t['profit_loss'] > 0])
            avg_loss = np.mean([abs(t['profit_loss']) for t in closed_trades if t['profit_loss'] < 0])
            profit_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        else:
            profit_ratio = 0
        
        # 夏普比率（年化）
        if daily_returns:
            returns_array = np.array(daily_returns)
            sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        portfolio_values_array = [pv['value'] for pv in portfolio_values]
        peak = np.maximum.accumulate(portfolio_values_array)
        drawdown = (peak - portfolio_values_array) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Calmar比率
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # 索提诺比率
        if daily_returns:
            negative_returns = [r for r in daily_returns if r < 0]
            downside_std = np.std(negative_returns) if negative_returns else 0
            sortino_ratio = np.mean(daily_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        else:
            sortino_ratio = 0
        
        return {
            'total_return': total_return,
            'annualized_return': total_return * (252 / len(daily_returns)) if daily_returns else 0,
            'win_rate': win_rate,
            'profit_ratio': profit_ratio,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'total_trades': len(closed_trades),
            'winning_trades': winning_trades,
            'losing_trades': len(closed_trades) - winning_trades,
            'avg_trade_return': np.mean([t.get('profit_loss_pct', 0) for t in closed_trades]) if closed_trades else 0
        }
    
    def generate_comprehensive_report(self):
        """生成综合回测报告"""
        print(f"\n{'='*80}")
        print(f"📊 自适应复利引擎 - 2024年全面回测报告")
        print(f"{'='*80}")
        
        for symbol, result in self.results.items():
            metrics = result['metrics']
            
            print(f"\n🎯 {symbol} 绩效分析:")
            print(f"  📈 绝对收益: {metrics['total_return']:+.2%}")
            print(f"  📊 年化收益: {metrics['annualized_return']:+.2%}")
            print(f"  🎯 胜率: {metrics['win_rate']:.1%}")
            print(f"  ⚖️  平均盈亏比: {metrics['profit_ratio']:.2f}")
            print(f"  📉 最大回撤: {metrics['max_drawdown']:.2%}")
            print(f"  🌟 夏普比率: {metrics['sharpe_ratio']:.2f}")
            print(f"  🚀 索提诺比率: {metrics['sortino_ratio']:.2f}")
            print(f"  🔄 Calmar比率: {metrics['calmar_ratio']:.2f}")
            print(f"  🔢 总交易次数: {metrics['total_trades']}")
            print(f"  ✅ 盈利交易: {metrics['winning_trades']}")
            print(f"  ❌ 亏损交易: {metrics['losing_trades']}")
            print(f"  💰 平均交易收益: {metrics['avg_trade_return']:+.2f}%")
            
            # 显示最近交易
            recent_trades = result['trade_history'][-5:]
            if recent_trades:
                print(f"\n  📋 最近5笔交易:")
                for trade in recent_trades:
                    if trade['action'] == 'SELL':
                        pnl = trade.get('profit_loss_pct', 0)
                        status = "盈利" if pnl > 0 else "亏损"
                        print(f"     {trade['date'].strftime('%Y-%m-%d')}: {trade['action']} | "
                              f"收益率: {pnl:+.2f}% ({status})")
                    else:
                        print(f"     {trade['date'].strftime('%Y-%m-%d')}: {trade['action']}")
        
        # 汇总统计
        if self.results:
            print(f"\n{'='*50}")
            print(f"📈 组合汇总统计")
            print(f"{'='*50}")
            
            avg_return = np.mean([r['metrics']['total_return'] for r in self.results.values()])
            avg_sharpe = np.mean([r['metrics']['sharpe_ratio'] for r in self.results.values()])
            avg_win_rate = np.mean([r['metrics']['win_rate'] for r in self.results.values()])
            
            print(f"  平均收益率: {avg_return:+.2%}")
            print(f"  平均夏普比率: {avg_sharpe:.2f}")
            print(f"  平均胜率: {avg_win_rate:.1%}")
            print(f"  测试币种数: {len(self.results)}")

def run_comprehensive_backtest():
    """运行全面回测"""
    print("🚀 开始自适应复利引擎全面回测...")
    print("测试币种: BTC-USDT, ETH-USDT, DOGE-USDT")
    print("测试周期: 2024年全年 (365天)")
    print("初始资金: $10,000 per symbol")
    
    backtester = CompoundBacktestEngine(initial_capital=10000.0)
    
    # 测试三个主要币种
    symbols = ['BTC-USDT', 'ETH-USDT', 'DOGE-USDT']
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        backtester.run_backtest(symbol, periods=365)
    
    # 生成详细报告
    backtester.generate_comprehensive_report()
    
    return backtester

if __name__ == "__main__":
    backtester = run_comprehensive_backtest()