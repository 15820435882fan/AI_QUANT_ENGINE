# sniper_backtest_engine.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any
import asyncio

class SniperBacktestEngine:
    """刺客系统回测引擎 - 修复交易记录结构问题"""
    
    def __init__(self, initial_capital: float = 10000.0, leverage: int = 10, use_enhanced_detector: bool = True):
        # 首先设置日志
        self.setup_logging()
        
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.results = {}
        
        # 导入刺客组件 - 使用增强版检测器
        from sniper_signal_detector import SniperSignalDetector
        from sniper_position_manager import SniperPositionManager
        from enhanced_sniper_detector import EnhancedSniperDetector
        from advanced_position_manager import AdvancedPositionManager
        
        # 选择信号检测器
        if use_enhanced_detector:
            self.signal_detector = EnhancedSniperDetector()
            self.logger.info("使用增强版信号检测器")
        else:
            self.signal_detector = SniperSignalDetector()
            self.logger.info("使用基础版信号检测器")
            
        self.position_manager = SniperPositionManager(initial_capital)
        self.advanced_position_manager = AdvancedPositionManager(initial_capital)
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('SniperBacktest')

    def generate_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """生成指定时间段的历史测试数据"""
        # 为每个symbol和日期组合创建唯一随机种子
        seed_str = f"{symbol}_{start_date}"
        seed_value = hash(seed_str) % 10000
        np.random.seed(seed_value)
        
        # 基础价格（基于真实历史）
        base_prices = {
            'BTC/USDT': {'2024-01': 45000, '2024-08': 60000, '2024-10': 58000, '2024-06': 55000, '2024-12': 62000},
            'ETH/USDT': {'2024-01': 2500, '2024-08': 3200, '2024-10': 3000, '2024-06': 2800, '2024-12': 3300},
            'SOL/USDT': {'2024-01': 100, '2024-08': 150, '2024-10': 130, '2024-06': 120, '2024-12': 160},
            'ADA/USDT': {'2024-01': 0.4, '2024-08': 0.5, '2024-10': 0.45, '2024-06': 0.42, '2024-12': 0.52},
            'DOT/USDT': {'2024-01': 6.5, '2024-08': 8.0, '2024-10': 7.2, '2024-06': 6.8, '2024-12': 8.5},
            'AVAX/USDT': {'2024-01': 35, '2024-08': 45, '2024-10': 40, '2024-06': 38, '2024-12': 48},
            'LINK/USDT': {'2024-01': 15, '2024-08': 18, '2024-10': 16, '2024-06': 15.5, '2024-12': 19},
            'MATIC/USDT': {'2024-01': 0.75, '2024-08': 0.95, '2024-10': 0.85, '2024-06': 0.80, '2024-12': 1.0}
        }
        
        # 解析日期
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days = (end_dt - start_dt).days
        
        if days <= 0:
            days = 30  # 默认30天
        
        # 获取基础价格
        year_month = start_date[:7]  # 2024-01
        symbol_prices = base_prices.get(symbol, {})
        base_price = symbol_prices.get(year_month, 100)
        
        # 生成分钟级数据
        periods = days * 24 * 12  # 5分钟间隔
        dates = [start_dt + timedelta(minutes=5*i) for i in range(periods)]
        
        prices = [base_price]
        volumes = [np.random.randint(10000, 50000)]
        
        # 定义初始趋势和波动率
        trend = 0.0005
        volatility = 0.015
        
        # 模拟真实市场波动
        for i in range(1, periods):
            # 每周调整趋势（避免UnboundLocalError）
            if i % (7*24*12) == 0:
                trend = np.random.choice([-0.001, -0.0005, 0, 0.0005, 0.001])
                volatility = np.random.uniform(0.01, 0.025)
            
            # 随机事件（异常波动）
            event = 0
            volume_boost = 1.0
            if np.random.random() < 0.008:  # 0.8%概率异常
                event = np.random.normal(0, 0.08)
                volume_boost = np.random.uniform(2.5, 5.0)
            
            # 价格变化
            change = np.random.normal(trend, volatility) + event
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, base_price * 0.1))  # 防止归零
            
            # 成交量
            base_volume = np.random.randint(5000, 50000)
            volumes.append(base_volume * volume_boost)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        data.set_index('timestamp', inplace=True)
        return data
    
    def run_backtest(self, symbol: str, start_date: str, end_date: str, period_name: str = "", use_advanced_position: bool = True) -> Dict[str, Any]:
        """运行单个币种回测 - 修复交易记录结构问题"""
        period_desc = f"{period_name} ({start_date} 到 {end_date})" if period_name else f"{start_date} 到 {end_date}"
        self.logger.info(f"🎯 开始刺客回测: {symbol} - {period_desc}")
        
        # 生成历史数据
        historical_data = self.generate_historical_data(symbol, start_date, end_date)
        
        # 回测参数
        capital = self.initial_capital
        position = None
        trade_history = []
        portfolio_values = []
        daily_returns = []
        
        last_portfolio_value = capital
        last_daily_check = None
        
        # 模拟实时监控（每5分钟）
        for i in range(50, len(historical_data), 5):
            current_data = historical_data.iloc[:i]
            current_price = current_data['close'].iloc[-1]
            current_time = current_data.index[-1]
            
            # 每日收益率计算
            current_day = current_time.date()
            if last_daily_check != current_day and last_daily_check is not None:
                if portfolio_values:
                    daily_return = (portfolio_values[-1]['value'] - last_portfolio_value) / last_portfolio_value
                    daily_returns.append(daily_return)
                    last_portfolio_value = portfolio_values[-1]['value']
            last_daily_check = current_day
            
            try:
                # 模拟监控异常波动
                if len(current_data) >= 20:
                    current_volume = current_data['volume'].iloc[-1]
                    avg_volume = current_data['volume'].tail(20).mean()
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                    
                    if len(current_data) > 1:
                        price_change = (current_data['close'].iloc[-1] - current_data['close'].iloc[-2]) / current_data['close'].iloc[-2]
                    else:
                        price_change = 0
                    
                    # 检查异常条件
                    volume_threshold = getattr(self.signal_detector, 'volume_threshold', 2.5)
                    price_threshold = getattr(self.signal_detector, 'price_threshold', 0.02)
                    
                    if volume_ratio > volume_threshold and abs(price_change) > price_threshold:
                        alert = {
                            'exchange': 'binance',
                            'symbol': symbol,
                            'volume_ratio': volume_ratio,
                            'price_change': price_change,
                            'current_price': current_price,
                            'timestamp': current_time
                        }
                        
                        # 信号确认 - 支持多空
                        signal = self.signal_detector.confirm_sniper_signal(alert, current_data)
                        
                        if signal['confirmed'] and position is None:
                            # 设置杠杆
                            signal['leverage'] = self.leverage
                            
                            # 选择仓位管理器
                            if use_advanced_position:
                                # 计算市场条件
                                market_conditions = {
                                    'volatility': np.std(np.diff(current_data['close'].tail(20)) / current_data['close'].tail(19)) if len(current_data) > 20 else 0.02
                                }
                                position_info = self.advanced_position_manager.calculate_dynamic_position(signal, market_conditions)
                            else:
                                position_info = self.position_manager.calculate_position_size(signal)
                            
                            # 创建开仓记录 - 统一添加action字段
                            trade = {
                                'action': 'OPEN',  # 添加action字段
                                'entry_time': current_time,
                                'symbol': symbol,
                                'direction': signal['direction'],  # LONG 或 SHORT
                                'entry_price': signal['entry_price'],
                                'position_size': position_info['position_size'],
                                'leverage': position_info['leverage'],
                                'quantity': position_info['quantity'],
                                'stop_loss': position_info['stop_loss'],
                                'take_profit': position_info['take_profit'],
                                'confidence': signal['confidence'],
                                'status': 'OPEN',
                                'volume_ratio': volume_ratio,
                                'price_change': price_change,
                                'technical_score': signal.get('technical_score', {})
                            }
                            
                            capital -= position_info['position_size']  # 冻结资金
                            position = trade
                            trade_history.append(trade)
                            
                            self.logger.info(f"🎯 {signal['direction']}开仓: {symbol} @ {signal['entry_price']:.2f} "
                                          f"杠杆: {position_info['leverage']}x 仓位: ${position_info['position_size']:.0f} "
                                          f"置信度: {signal['confidence']:.1%}")
                
                # 检查平仓条件（支持多空）
                if position:
                    # 计算盈亏 - 必须在try块内部
                    pnl = self._calculate_pnl(position, current_price)
                    
                    # 止损检查
                    stop_loss_triggered = False
                    take_profit_triggered = False
                    
                    if position['direction'] == 'LONG':
                        stop_loss_triggered = current_price <= position['stop_loss']
                        take_profit_triggered = current_price >= position['take_profit']
                    else:  # SHORT
                        stop_loss_triggered = current_price >= position['stop_loss']
                        take_profit_triggered = current_price <= position['take_profit']
                    
                    if stop_loss_triggered or take_profit_triggered:
                        # 平仓
                        reason = 'STOP_LOSS' if stop_loss_triggered else 'TAKE_PROFIT'
                        capital += position['position_size'] + pnl  # 解冻资金 + 盈亏
                        
                        # 计算持仓时间
                        entry_time = position['entry_time']
                        if isinstance(entry_time, str):
                            entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                        
                        holding_period = current_time - entry_time
                        holding_hours = holding_period.total_seconds() / 3600
                        
                        # 创建平仓交易记录 - 统一结构
                        close_trade = {
                            'action': 'CLOSE',
                            'symbol': symbol,
                            'direction': position['direction'],
                            'entry_time': position['entry_time'],
                            'entry_price': position['entry_price'],
                            'exit_time': current_time,
                            'exit_price': current_price,
                            'position_size': position['position_size'],
                            'leverage': position['leverage'],
                            'quantity': position['quantity'],
                            'pnl': pnl,
                            'pnl_pct': (pnl / position['position_size']) * 100,
                            'holding_hours': holding_hours,
                            'reason': reason,
                            'confidence': position['confidence'],
                            'volume_ratio': position.get('volume_ratio', 0),
                            'price_change': position.get('price_change', 0)
                        }
                        
                        trade_history.append(close_trade)
                        
                        status = "盈利" if pnl > 0 else "亏损"
                        self.logger.info(f"💸 平仓: {position['direction']} {symbol} | "
                                      f"{reason} | {status}: ${pnl:+.0f} ({pnl/position['position_size']*100:+.1f}%) | "
                                      f"持仓: {holding_hours:.1f}小时")
                        position = None
            
            except Exception as e:
                self.logger.error(f"回测过程错误: {e}")
                continue
            
            # 记录组合价值
            current_portfolio_value = capital + (position['position_size'] + self._calculate_pnl(position, current_price) if position else 0)
            portfolio_values.append({
                'timestamp': current_time,
                'value': current_portfolio_value
            })
        
        # 最终日收益率计算
        if portfolio_values and last_portfolio_value > 0:
            final_return = (portfolio_values[-1]['value'] - last_portfolio_value) / last_portfolio_value
            daily_returns.append(final_return)
        
        # 计算绩效指标
        metrics = self._calculate_performance_metrics(trade_history, portfolio_values, daily_returns)
        
        result_key = f"{symbol}_{period_name}" if period_name else symbol
        self.results[result_key] = {
            'period': period_desc,
            'metrics': metrics,
            'trade_history': trade_history,
            'portfolio_values': portfolio_values
        }
        
        return self.results[result_key]
    
    def _calculate_pnl(self, position: Dict, current_price: float) -> float:
        """计算盈亏（支持多空）"""
        if not position:
            return 0
            
        quantity = position['quantity']
        leverage = position['leverage']
        entry_price = position['entry_price']
        
        if position['direction'] == 'LONG':
            return (current_price - entry_price) * quantity * leverage
        else:  # SHORT
            return (entry_price - current_price) * quantity * leverage
    
    def _calculate_performance_metrics(self, trade_history: List, portfolio_values: List, daily_returns: List) -> Dict[str, float]:
        """计算完整的绩效指标 - 修复交易记录过滤"""
        # 只处理平仓交易
        closed_trades = [t for t in trade_history if t.get('action') == 'CLOSE']
        
        if not closed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0,
                'avg_trade_return': 0,
                'avg_holding_hours': 0
            }
        
        # 基础指标
        total_trades = len(closed_trades)
        winning_trades = len([t for t in closed_trades if t['pnl'] > 0])
        win_rate = winning_trades / total_trades
        
        # 盈亏统计
        profits = [t['pnl'] for t in closed_trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in closed_trades if t['pnl'] < 0]
        
        total_profit = sum(profits) if profits else 0
        total_loss = abs(sum(losses)) if losses else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # 持仓时间统计
        holding_hours = [t.get('holding_hours', 0) for t in closed_trades]
        avg_holding_hours = np.mean(holding_hours) if holding_hours else 0
        
        # 总收益
        total_pnl = sum(t['pnl'] for t in closed_trades)
        total_return = total_pnl / self.initial_capital
        
        # 夏普比率（年化）
        if daily_returns:
            returns_array = np.array(daily_returns)
            sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        portfolio_values_array = [pv['value'] for pv in portfolio_values]
        if portfolio_values_array:
            peak = np.maximum.accumulate(portfolio_values_array)
            drawdown = (peak - portfolio_values_array) / peak
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        else:
            max_drawdown = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'avg_trade_return': total_pnl / total_trades if total_trades > 0 else 0,
            'avg_winning_trade': avg_profit,
            'avg_losing_trade': avg_loss,
            'avg_holding_hours': avg_holding_hours
        }
    
    def generate_comprehensive_report(self):
        """生成综合回测报告"""
        print(f"\n{'='*100}")
        print(f"🎯 刺客交易系统 - 综合回测性能报告")
        print(f"{'='*100}")
        print(f"初始资金: ${self.initial_capital:,.2f} | 杠杆: {self.leverage}x")
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"信号检测器: {'增强版' if hasattr(self.signal_detector, 'volume_threshold') else '基础版'}")
        print(f"{'='*100}")
        
        # 按时间段汇总
        period_results = {}
        for key, result in self.results.items():
            symbol = key.split('_')[0]
            period = key.split('_')[1] if '_' in key else 'Overall'
            
            if period not in period_results:
                period_results[period] = []
            period_results[period].append(result)
        
        # 按时间段显示结果
        for period, results in period_results.items():
            print(f"\n📅 时间段: {period}")
            print(f"{'-'*80}")
            
            total_pnl = 0
            total_trades = 0
            winning_trades = 0
            
            for result in results:
                metrics = result['metrics']
                symbol = result['period'].split(' ')[0]
                
                total_pnl += metrics['total_pnl']
                total_trades += metrics['total_trades']
                winning_trades += metrics['winning_trades']
                
                print(f"  {symbol:12} | "
                      f"交易: {metrics['total_trades']:2d} | "
                      f"胜率: {metrics['win_rate']:6.1%} | "
                      f"收益: ${metrics['total_pnl']:7.0f} | "
                      f"夏普: {metrics['sharpe_ratio']:5.2f} | "
                      f"回撤: {metrics['max_drawdown']:6.1%} | "
                      f"持仓: {metrics['avg_holding_hours']:5.1f}h")
            
            # 时间段汇总
            if total_trades > 0:
                period_win_rate = winning_trades / total_trades
                print(f"  {'汇总':12} | "
                      f"交易: {total_trades:2d} | "
                      f"胜率: {period_win_rate:6.1%} | "
                      f"收益: ${total_pnl:7.0f} | "
                      f"总收益率: {total_pnl/self.initial_capital:6.1%}")

def run_detailed_analysis():
    """运行详细交易分析"""
    print("🚀 启动刺客交易系统详细分析...")
    
    from trade_analyzer import TradeAnalyzer
    
    # 使用增强版检测器和高级仓位管理
    backtester = SniperBacktestEngine(
        initial_capital=10000.0, 
        leverage=10, 
        use_enhanced_detector=True
    )
    
    analyzer = TradeAnalyzer()
    
    # 重点测试表现最好的币种和时间段
    test_cases = [
        ('SOL/USDT', '2024-01-01', '2024-01-31', '2024年1月'),
        ('ADA/USDT', '2024-01-01', '2024-01-31', '2024年1月'),
        ('SOL/USDT', '2024-01-01', '2024-06-30', '2024上半年'),
        ('ADA/USDT', '2024-01-01', '2024-06-30', '2024上半年'),
        ('BTC/USDT', '2024-01-01', '2024-01-31', '2024年1月'),  # 对比测试
    ]
    
    for symbol, start_date, end_date, period_name in test_cases:
        print(f"\n{'='*80}")
        print(f"🎯 详细分析: {symbol} - {period_name}")
        print(f"{'='*80}")
        
        result = backtester.run_backtest(
            symbol, start_date, end_date, period_name, 
            use_advanced_position=True
        )
        
        # 生成详细交易报告
        analyzer.generate_trade_report(result['trade_history'], symbol, period_name)
    
    # 生成综合报告
    backtester.generate_comprehensive_report()
    
    return backtester, analyzer

def run_comprehensive_backtest():
    """运行综合回测"""
    print("🚀 启动刺客交易系统综合回测...")
    print("测试币种: BTC, ETH, SOL, ADA, DOT, AVAX, LINK, MATIC")
    print("杠杆: 10x | 初始资金: $10,000")
    print("信号检测器: 增强版 | 仓位管理: 高级动态")
    print("\n📅 测试时间段:")
    print("  - 2024年1月 (市场筑底期)")
    print("  - 2024年8月 (夏季行情)") 
    print("  - 2024年10月 (秋季波动)")
    print("  - 2024年上半年 (1月-6月)")
    print("  - 2024年全年 (1月-12月)")
    
    backtester = SniperBacktestEngine(
        initial_capital=10000.0, 
        leverage=10, 
        use_enhanced_detector=True
    )
    
    # 测试币种
    test_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT', 'AVAX/USDT', 'LINK/USDT', 'MATIC/USDT']
    
    # 多时间段回测
    test_periods = [
        ('2024-01', '2024-01-01', '2024-01-31', '1月'),
        ('2024-08', '2024-08-01', '2024-08-31', '8月'),
        ('2024-10', '2024-10-01', '2024-10-31', '10月'),
        ('2024-H1', '2024-01-01', '2024-06-30', '上半年'),
        ('2024-Full', '2024-01-01', '2024-12-31', '全年')
    ]
    
    for period_name, start_date, end_date, desc in test_periods:
        print(f"\n{'='*60}")
        print(f"📊 测试时间段: {desc}")
        print(f"{'='*60}")
        
        for symbol in test_symbols:
            backtester.run_backtest(
                symbol, start_date, end_date, period_name, 
                use_advanced_position=True
            )
    
    # 生成详细报告
    backtester.generate_comprehensive_report()
    
    return backtester

if __name__ == "__main__":
    # 选择运行模式
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--detailed':
        # 详细分析模式
        backtester, analyzer = run_detailed_analysis()
    else:
        # 综合回测模式
        backtester = run_comprehensive_backtest()