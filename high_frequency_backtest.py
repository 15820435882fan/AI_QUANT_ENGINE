# high_frequency_backtest.py (修复类作用域问题)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any

class OptimizedCapitalManager:
    """优化版资金管理器 - 移到外部类"""
    
    def __init__(self, capital):
        self.total_capital = capital
        self.available_capital = capital
        self.active_positions = {}
        self.total_positions = 0
        self.total_pnl = 0
        self.liquidated_positions = 0
        self.max_drawdown = 0
        self.peak_capital = capital
    
    def calculate_position_size(self, symbol, signal, is_main):
        if symbol in self.active_positions:
            return {'position_size': 0, 'error': 'Position exists'}
        
        # 动态仓位计算（考虑爆仓风险）
        if is_main:
            position_size = min(1000, self.available_capital * 0.1)
            position_size = max(position_size, 200)
        else:
            position_size = min(500, self.available_capital * 0.05)
            position_size = max(position_size, 100)
        
        # 计算爆仓价格
        entry_price = signal['entry_price']
        leverage = signal.get('leverage', 10)
        direction = signal['signal']
        
        if direction == 'LONG':
            liquidation_price = entry_price * (1 - 1/leverage * 0.9)  # 90%保证金率
        else:
            liquidation_price = entry_price * (1 + 1/leverage * 0.9)
        
        return {
            'position_size': position_size,
            'leverage': leverage,
            'quantity': position_size / entry_price,
            'liquidation_price': liquidation_price,
            'is_main_symbol': is_main
        }
    
    def open_position(self, symbol, position_info):
        position_size = position_info['position_size']
        
        if position_size > 0 and position_size <= self.available_capital:
            self.active_positions[symbol] = position_info
            self.available_capital -= position_size
            self.total_positions += 1
    
    def close_position(self, symbol, pnl, reason='NORMAL'):
        if symbol in self.active_positions:
            position = self.active_positions[symbol]
            position_size = position['position_size']
            
            self.available_capital += position_size + pnl
            self.total_capital += pnl
            self.total_pnl += pnl
            
            # 更新最大回撤
            if self.total_capital > self.peak_capital:
                self.peak_capital = self.total_capital
            else:
                drawdown = (self.peak_capital - self.total_capital) / self.peak_capital
                self.max_drawdown = max(self.max_drawdown, drawdown)
            
            # 统计爆仓
            if reason == 'LIQUIDATION':
                self.liquidated_positions += 1
            
            del self.active_positions[symbol]
    
    def check_liquidation(self, symbol, current_price):
        """检查是否爆仓"""
        if symbol in self.active_positions:
            position = self.active_positions[symbol]
            liquidation_price = position.get('liquidation_price')
            
            if liquidation_price:
                if (position['direction'] == 'LONG' and current_price <= liquidation_price) or \
                   (position['direction'] == 'SHORT' and current_price >= liquidation_price):
                    # 计算爆仓损失（损失全部仓位资金）
                    pnl = -position['position_size']  # 损失全部仓位价值
                    self.close_position(symbol, pnl, 'LIQUIDATION')
                    return True
        return False
    
    def get_portfolio_status(self):
        total_position_value = sum(pos['position_size'] for pos in self.active_positions.values())
        
        return {
            'total_capital': self.total_capital,
            'available_capital': self.available_capital,
            'used_capital': total_position_value,
            'active_positions': len(self.active_positions),
            'total_positions': self.total_positions,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'liquidated_positions': self.liquidated_positions,
            'utilization_rate': total_position_value / self.total_capital if self.total_capital > 0 else 0
        }

class OptimizedStrategy:
    """优化版策略 - 移到外部类"""
    
    def __init__(self):
        self.leverage = 10
    
    def detect_opportunity(self, symbol, df, timeframe='5min'):
        if len(df) < 20:
            return {'signal': 'HOLD'}
        
        current_price = df['close'].iloc[-1]
        
        if len(df) >= 10:
            volume_ratio = df['volume'].iloc[-1] / df['volume'].tail(10).mean()
            price_change = (current_price - df['close'].iloc[-5]) / df['close'].iloc[-5] if len(df) >= 5 else 0
            
            if volume_ratio > 2.0 and abs(price_change) > 0.01 and np.random.random() < 0.3:
                direction = 'LONG' if price_change > 0 else 'SHORT'
                
                if direction == 'LONG':
                    stop_loss = current_price * 0.98
                    take_profit = current_price * 1.05
                else:
                    stop_loss = current_price * 1.02
                    take_profit = current_price * 0.95
                
                return {
                    'signal': direction,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'leverage': self.leverage,
                    'confidence': min(0.8, 0.5 + abs(price_change) * 10),
                    'volume_ratio': volume_ratio
                }
        
        return {'signal': 'HOLD'}

class OptimizedAnalyzer:
    """优化版分析器 - 移到外部类"""
    
    def __init__(self):
        self.results = {}
        self.yearly_stats = {}
    
    def analyze_trade_results(self, trades, symbol, period, capital_info=None):
        closed_trades = [t for t in trades if t.get('action') == 'CLOSE']
        
        if closed_trades:
            winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in closed_trades if t.get('pnl', 0) < 0]
            liquidated_trades = [t for t in closed_trades if t.get('reason') == 'LIQUIDATION']
            
            total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
            win_rate = len(winning_trades) / len(closed_trades)
            liquidation_rate = len(liquidated_trades) / len(closed_trades)
            
            # 月度统计
            monthly_stats = {}
            for trade in closed_trades:
                month_key = trade.get('exit_time', datetime.now()).strftime('%Y-%m')
                if month_key not in monthly_stats:
                    monthly_stats[month_key] = {'trades': 0, 'pnl': 0, 'wins': 0, 'liquidations': 0}
                monthly_stats[month_key]['trades'] += 1
                monthly_stats[month_key]['pnl'] += trade.get('pnl', 0)
                if trade.get('pnl', 0) > 0:
                    monthly_stats[month_key]['wins'] += 1
                if trade.get('reason') == 'LIQUIDATION':
                    monthly_stats[month_key]['liquidations'] += 1
            
            for month in monthly_stats:
                stats = monthly_stats[month]
                stats['win_rate'] = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
                stats['liquidation_rate'] = stats['liquidations'] / stats['trades'] if stats['trades'] > 0 else 0
        else:
            closed_trades = []
            total_pnl = 0
            win_rate = 0
            liquidation_rate = 0
            monthly_stats = {}
        
        result = {
            'period': period,
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades) if closed_trades else 0,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'liquidation_rate': liquidation_rate,
            'monthly_stats': monthly_stats,
            'final_capital': capital_info.get('final_capital', 0) if capital_info else 0,
            'max_drawdown': capital_info.get('max_drawdown', 0) if capital_info else 0
        }
        
        self.results[symbol] = result
    
    def generate_detailed_report(self):
        print(f"\n{'='*120}")
        print(f"🎯 高频交易系统 - 完整回测报告 (含爆仓风险和复利)")
        print(f"{'='*120}")
        
        total_trades = 0
        total_pnl = 0
        total_liquidations = 0
        symbols_with_trades = 0
        
        print(f"\n{'币种':<12} {'周期':<8} {'交易数':<6} {'胜率':<6} {'爆仓率':<6} {'总收益':<10} {'最大回撤':<8}")
        print(f"{'-'*80}")
        
        for symbol, result in self.results.items():
            if result['total_trades'] > 0:
                total_trades += result['total_trades']
                total_pnl += result['total_pnl']
                total_liquidations += result['total_trades'] * result['liquidation_rate']
                symbols_with_trades += 1
                
                print(f"{symbol:<12} {result['period']:<8} {result['total_trades']:<6} "
                      f"{result['win_rate']:<6.1%} {result['liquidation_rate']:<6.1%} "
                      f"${result['total_pnl']:<9.0f} {result['max_drawdown']:<7.1%}")
        
        if symbols_with_trades > 0:
            avg_win_rate = sum(r['win_rate'] for r in self.results.values() if r['total_trades'] > 0) / symbols_with_trades
            avg_liquidation_rate = total_liquidations / total_trades if total_trades > 0 else 0
            
            print(f"\n{'='*80}")
            print(f"📈 系统汇总统计:")
            print(f"  📊 总交易次数: {total_trades}笔")
            print(f"  🎯 平均胜率: {avg_win_rate:.1%}")
            print(f"  ⚠️  平均爆仓率: {avg_liquidation_rate:.1%}")
            print(f"  💰 总收益: ${total_pnl:+,.0f}")
            print(f"  🌐 测试币种: {symbols_with_trades}个")

class HighFrequencyBacktestEngine:
    """高频交易回测引擎 - 修复版"""
    
    def __init__(self, initial_capital: float = 10000.0, use_compounding: bool = True):
        self.setup_logging()
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.use_compounding = use_compounding
        self.results = {}
        self.monthly_results = {}
        
        # 使用外部类
        self.strategy = OptimizedStrategy()
        self.capital_manager = OptimizedCapitalManager(self.current_capital)
        self.analyzer = OptimizedAnalyzer()
        
        # 交易对配置
        self.main_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']
        self.small_symbols = [
            'ADA/USDT', 'DOT/USDT', 'AVAX/USDT', 'LINK/USDT', 'MATIC/USDT',
            'ATOM/USDT', 'NEAR/USDT', 'FTM/USDT', 'ALGO/USDT', 'SAND/USDT'
        ]
        
        self.logger.info("🚀 高频交易系统初始化完成")
        self.logger.info(f"💰 初始资金: ${initial_capital:,.0f}")
        self.logger.info(f"📈 复利模式: {'开启' if use_compounding else '关闭'}")
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('HighFrequencyBacktest')
    
    def generate_market_data(self, symbol: str, start_date: str, end_date: str, timeframe: str = '5min') -> pd.DataFrame:
        """生成市场数据"""
        seed_str = f"{symbol}_{start_date}"
        seed_value = hash(seed_str) % 10000
        np.random.seed(seed_value)
        
        base_prices = {
            'BTC/USDT': 45000, 'ETH/USDT': 2500, 'SOL/USDT': 100, 'BNB/USDT': 300,
            'ADA/USDT': 0.4, 'DOT/USDT': 6.5, 'AVAX/USDT': 35, 'LINK/USDT': 15,
            'MATIC/USDT': 0.75, 'ATOM/USDT': 10, 'NEAR/USDT': 3, 'FTM/USDT': 0.3,
            'ALGO/USDT': 0.2, 'SAND/USDT': 0.5
        }
        
        base_price = base_prices.get(symbol, 10)
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days = max(1, (end_dt - start_dt).days)
        
        # 生成5分钟数据
        periods = days * 24 * 12
        dates = pd.date_range(start=start_dt, periods=periods, freq='5T')
        
        prices = [base_price]
        volumes = [np.random.randint(10000, 50000)]
        
        # 市场参数
        trend = np.random.normal(0.0002, 0.0003)
        base_volatility = 0.006
        
        for i in range(1, periods):
            # 创造交易机会
            event = 0
            volume_boost = 1.0
            
            if np.random.random() < 0.15:
                event = np.random.normal(0, 0.02)
                volume_boost = np.random.uniform(2.5, 6.0)
            
            change = np.random.normal(trend, base_volatility) + event
            new_price = max(prices[-1] * (1 + change), base_price * 0.1)
            prices.append(new_price)
            
            base_volume = np.random.randint(8000, 25000)
            volumes.append(int(base_volume * volume_boost))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.004))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.004))) for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        data.set_index('timestamp', inplace=True)
        return data
    
    def run_symbol_backtest(self, symbol: str, start_date: str, end_date: str, 
                          period_name: str = "", is_main_symbol: bool = False) -> Dict[str, Any]:
        """运行单个币种回测"""
        self.logger.info(f"🎯 开始回测: {symbol} - {period_name}")
        
        market_data = self.generate_market_data(symbol, start_date, end_date, '5min')
        trade_history = []
        current_positions = {}
        
        step_size = max(1, len(market_data) // 500)
        
        for i in range(20, len(market_data), step_size):
            current_data = market_data.iloc[:i]
            current_price = current_data['close'].iloc[-1]
            current_time = current_data.index[-1]
            
            try:
                # 1. 检查爆仓
                if symbol in current_positions:
                    if self.capital_manager.check_liquidation(symbol, current_price):
                        self.logger.warning(f"💥 {symbol} 爆仓！当前价格: {current_price:.4f}")
                        continue
                
                # 2. 检测交易机会
                signal = self.strategy.detect_opportunity(symbol, current_data, '5min')
                
                # 3. 开仓逻辑
                if signal['signal'] in ['LONG', 'SHORT'] and symbol not in current_positions:
                    position_info = self.capital_manager.calculate_position_size(
                        symbol, signal, is_main_symbol
                    )
                    
                    if position_info['position_size'] > 0:
                        trade = {
                            'action': 'OPEN',
                            'symbol': symbol,
                            'direction': signal['signal'],
                            'entry_time': current_time,
                            'entry_price': signal['entry_price'],
                            'position_size': position_info['position_size'],
                            'leverage': signal['leverage'],
                            'quantity': position_info['quantity'],
                            'stop_loss': signal['stop_loss'],
                            'take_profit': signal['take_profit'],
                            'liquidation_price': position_info['liquidation_price'],
                            'is_main_symbol': is_main_symbol
                        }
                        
                        self.capital_manager.open_position(symbol, trade)
                        current_positions[symbol] = trade
                        trade_history.append(trade)
                
                # 4. 平仓逻辑
                if symbol in current_positions:
                    position = current_positions[symbol]
                    pnl = self._calculate_pnl(position, current_price)
                    
                    stop_loss_triggered = False
                    take_profit_triggered = False
                    
                    if position['direction'] == 'LONG':
                        stop_loss_triggered = current_price <= position['stop_loss']
                        take_profit_triggered = current_price >= position['take_profit']
                    else:
                        stop_loss_triggered = current_price >= position['stop_loss']
                        take_profit_triggered = current_price <= position['take_profit']
                    
                    if stop_loss_triggered or take_profit_triggered:
                        reason = 'STOP_LOSS' if stop_loss_triggered else 'TAKE_PROFIT'
                        self.capital_manager.close_position(symbol, pnl, reason)
                        
                        close_trade = {
                            'action': 'CLOSE',
                            'symbol': symbol,
                            'exit_time': current_time,
                            'exit_price': current_price,
                            'pnl': pnl,
                            'reason': reason
                        }
                        trade_history.append(close_trade)
                        del current_positions[symbol]
            
            except Exception as e:
                self.logger.error(f"交易过程错误 {symbol}: {e}")
                continue
        
        # 分析交易结果
        portfolio_status = self.capital_manager.get_portfolio_status()
        capital_info = {
            'final_capital': portfolio_status['total_capital'],
            'max_drawdown': portfolio_status['max_drawdown']
        }
        
        self.analyzer.analyze_trade_results(trade_history, symbol, period_name, capital_info)
        
        return {
            'symbol': symbol,
            'period': period_name,
            'trade_history': trade_history,
            'final_capital': portfolio_status['total_capital'],
            'max_drawdown': portfolio_status['max_drawdown'],
            'liquidated_positions': portfolio_status['liquidated_positions']
        }
    
    def _calculate_pnl(self, position: Dict, current_price: float) -> float:
        quantity = position['quantity']
        leverage = position['leverage']
        entry_price = position['entry_price']
        
        if position['direction'] == 'LONG':
            return (current_price - entry_price) * quantity * leverage
        else:
            return (entry_price - current_price) * quantity * leverage
    
    def run_comprehensive_backtest(self, test_periods: List = None):
        """运行综合回测"""
        if test_periods is None:
            # 2025年测试数据
            test_periods = [
                ('2025-01', '2025-01-01', '2025-01-31', '1月'),
                ('2025-02', '2025-02-01', '2025-02-28', '2月'),
                ('2025-03', '2025-03-01', '2025-03-31', '3月'),
                ('2025-04', '2025-04-01', '2025-04-30', '4月'),
                ('2025-05', '2025-05-01', '2025-05-31', '5月'),
                ('2025-06', '2025-06-01', '2025-06-30', '6月'),
                ('2025-07', '2025-07-01', '2025-07-31', '7月'),
                ('2025-08', '2025-08-01', '2025-08-31', '8月'),
                ('2025-09', '2025-09-01', '2025-09-30', '9月'),
                ('2025-10', '2025-10-01', '2025-10-31', '10月'),
                ('2025-11', '2025-11-01', '2025-11-20', '11月(前20天)'),
            ]
        
        self.logger.info("🚀 开始2025年高频交易系统回测")
        self.logger.info(f"📅 测试周期: 2025年1月-11月")
        self.logger.info(f"💰 初始资金: ${self.initial_capital:,.0f}")
        
        monthly_capital = self.initial_capital
        yearly_results = []
        
        for period_id, start_date, end_date, period_name in test_periods:
            self.logger.info(f"\n{'📅'*20} {period_name} {'📅'*20}")
            
            # 复利机制：每月初重置资金
            if self.use_compounding:
                self.current_capital = monthly_capital
                self.capital_manager = OptimizedCapitalManager(self.current_capital)
                self.logger.info(f"💰 本月初始资金: ${monthly_capital:,.0f}")
            
            # 测试币种
            test_symbols = self.main_symbols + self.small_symbols[:5]
            monthly_pnl = 0
            
            for symbol in test_symbols:
                is_main = symbol in self.main_symbols
                result = self.run_symbol_backtest(symbol, start_date, end_date, period_name, is_main)
                monthly_pnl += result.get('total_pnl', 0)
            
            # 更新月度资金（复利）
            if self.use_compounding:
                monthly_capital += monthly_pnl
                self.logger.info(f"📈 {period_name}收益: ${monthly_pnl:+,.0f}, 月末资金: ${monthly_capital:,.0f}")
            
            yearly_results.append({
                'month': period_name,
                'starting_capital': self.current_capital,
                'monthly_pnl': monthly_pnl,
                'ending_capital': monthly_capital
            })
        
        # 生成详细报告
        self.analyzer.generate_detailed_report()
        
        # 年度总结
        final_capital = monthly_capital if self.use_compounding else self.capital_manager.total_capital
        total_return = final_capital - self.initial_capital
        annual_return_pct = (total_return / self.initial_capital) * 100
        
        portfolio_status = self.capital_manager.get_portfolio_status()
        
        self.logger.info(f"\n{'💰'*30} 2025年度报告 {'💰'*30}")
        self.logger.info(f"💰 初始资金: ${self.initial_capital:,.0f}")
        self.logger.info(f"💰 最终资金: ${final_capital:,.0f}")
        self.logger.info(f"📈 年度收益: ${total_return:+,.0f}")
        self.logger.info(f"🎯 年化收益率: {annual_return_pct:+.1f}%")
        self.logger.info(f"📊 总交易次数: {portfolio_status['total_positions']}次")
        self.logger.info(f"⚡ 最大回撤: {portfolio_status['max_drawdown']:.1%}")
        self.logger.info(f"💥 爆仓次数: {portfolio_status['liquidated_positions']}次")
        self.logger.info(f"📈 复利模式: {'开启' if self.use_compounding else '关闭'}")
        
        # 显示月度资金变化
        if self.use_compounding:
            self.logger.info(f"\n{'📊'*10} 月度资金变化 {'📊'*10}")
            for result in yearly_results:
                self.logger.info(f"  {result['month']}: ${result['starting_capital']:,.0f} → ${result['ending_capital']:,.0f} "
                              f"(收益: ${result['monthly_pnl']:+,.0f})")
        
        return yearly_results

def run_high_frequency_backtest():
    """运行高频交易回测"""
    print("🚀 启动2025年高频交易系统回测...")
    print("=" * 80)
    
    # 开启复利模式
    backtester = HighFrequencyBacktestEngine(initial_capital=10000.0, use_compounding=True)
    
    # 运行2025年回测
    results = backtester.run_comprehensive_backtest()
    
    print("\n" + "=" * 80)
    print("🎉 2025年回测完成！")
    print("=" * 80)
    
    return backtester, results

if __name__ == "__main__":
    backtester, results = run_high_frequency_backtest()