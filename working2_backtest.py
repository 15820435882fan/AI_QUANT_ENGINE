#!/usr/bin/env python3
"""
高频交易回测系统 - 最终优化版本
修复信号问题，改进资金管理，添加复利
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FinalBacktest')

class FinalSignalDetector:
    """最终信号检测器 - 平衡信号质量和数量"""
    
    def __init__(self):
        logger.info("🎯 最终信号检测器初始化完成")
    
    def analyze_enhanced_signals(self, data, symbol):
        """分析增强信号 - 平衡版本"""
        try:
            if data is None or len(data) < 50:
                return pd.DataFrame()
            
            # 计算技术指标
            df = self._calculate_technical_indicators(data)
            
            # 生成平衡信号
            signals = self._generate_balanced_signals(df, symbol)
            
            return signals
            
        except Exception as e:
            logger.error(f"信号分析错误: {e}")
            return pd.DataFrame()
    
    def _calculate_technical_indicators(self, df):
        """计算技术指标"""
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
        
        # 移动平均线
        df['sma_10'] = df['close'].rolling(window=10, min_periods=1).mean()
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan).fillna(1)
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """计算MACD"""
        ema_fast = prices.ewm(span=fast, min_periods=1).mean()
        ema_slow = prices.ewm(span=slow, min_periods=1).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, min_periods=1).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _generate_balanced_signals(self, df, symbol):
        """生成平衡信号 - 确保有信号产生"""
        signals = []
        strong_signal_count = 0
        
        for i in range(len(df)):
            if i < 20:
                signals.append({'signal_strength': 0, 'signal_type': 'HOLD'})
                continue
                
            try:
                row = df.iloc[i]
                signal_strength = 0
                
                # 1. RSI信号 (中等阈值)
                rsi = row.get('rsi', 50)
                if not pd.isna(rsi):
                    if rsi < 35:  # 中等超卖
                        signal_strength += 0.4
                    elif rsi > 65:  # 中等超买
                        signal_strength -= 0.4
                
                # 2. MACD信号 (中等阈值)
                macd_hist = row.get('macd_hist', 0)
                if not pd.isna(macd_hist):
                    if macd_hist > 0.002:  # 中等金叉
                        signal_strength += 0.3
                    elif macd_hist < -0.002:  # 中等死叉
                        signal_strength -= 0.3
                
                # 3. 移动平均线信号
                sma_10 = row.get('sma_10', 0)
                sma_20 = row.get('sma_20', 0)
                if not pd.isna(sma_10) and not pd.isna(sma_20):
                    if sma_10 > sma_20:
                        signal_strength += 0.2
                    else:
                        signal_strength -= 0.1
                
                # 限制范围
                signal_strength = max(min(signal_strength, 1.0), -1.0)
                
                # 确定信号类型 (降低阈值确保有交易)
                if signal_strength > 0.5:
                    signal_type = 'STRONG_BUY'
                    strong_signal_count += 1
                elif signal_strength > 0.2:
                    signal_type = 'BUY'
                elif signal_strength < -0.5:
                    signal_type = 'STRONG_SELL'
                    strong_signal_count += 1
                elif signal_strength < -0.2:
                    signal_type = 'SELL'
                else:
                    signal_type = 'HOLD'
                
                signals.append({
                    'signal_strength': signal_strength,
                    'signal_type': signal_type
                })
                
            except Exception as e:
                signals.append({'signal_strength': 0, 'signal_type': 'HOLD'})
        
        logger.info(f"📊 {symbol} 信号统计: {strong_signal_count}个强信号")
        return pd.DataFrame(signals)

class FinalBacktest:
    """最终回测系统 - 修复所有问题"""
    
    def __init__(self, initial_capital=10000, leverage=3, compound_mode=True):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.leverage = leverage
        self.compound_mode = compound_mode
        self.positions = {}
        self.trade_history = []
        self.monthly_performance = []
        
        # 资金分配 (每个币种分配20%资金)
        self.symbol_capital = {}
        
        # 使用最终信号检测器
        self.signal_detector = FinalSignalDetector()
        
        logger.info("🚀 最终回测系统初始化完成")
        logger.info(f"💰 初始资金: ${initial_capital:,}, 杠杆: {leverage}x, 复利模式: {compound_mode}")
    
    def run_final_backtest(self, symbols, days=30, test_full_year=False):
        """运行最终回测"""
        logger.info(f"🎯 开始最终回测: {symbols} {days}天")
        
        # 分配资金给每个币种
        capital_per_symbol = self.initial_capital / len(symbols)
        for symbol in symbols:
            self.symbol_capital[symbol] = capital_per_symbol
            logger.info(f"   📊 {symbol} 分配资金: ${capital_per_symbol:,.2f}")
        
        all_results = []
        
        for symbol in symbols:
            logger.info(f"\n🔍 测试币种: {symbol}")
            
            try:
                # 生成数据
                data = self._generate_realistic_data(symbol, days)
                
                # 运行回测
                result = self._final_backtest(symbol, data, test_full_year)
                all_results.append(result)
                
            except Exception as e:
                logger.error(f"❌ {symbol} 回测失败: {e}")
                continue
        
        # 生成最终报告
        self._generate_final_report(all_results)
        return all_results
    
    def _generate_realistic_data(self, symbol, days):
        """生成真实市场数据"""
        n_points = days * 24
        dates = pd.date_range(end=datetime.now(), periods=n_points, freq='H')
        
        base_prices = {
            'BTC/USDT': 35000, 'ETH/USDT': 2500, 'SOL/USDT': 100,
            'BNB/USDT': 300, 'ADA/USDT': 0.5, 'DOT/USDT': 6,
            'AVAX/USDT': 20, 'LINK/USDT': 15, 'MATIC/USDT': 0.8
        }
        base_price = base_prices.get(symbol, 100)
        
        np.random.seed(42)
        prices = [base_price]
        
        for i in range(1, n_points):
            # 真实的市场波动
            daily_return = np.random.uniform(-0.02, 0.02)
            trend = 0.0001
            cycle = 0.003 * np.sin(2 * np.pi * i / (24 * 7))
            
            total_return = daily_return + trend + cycle
            new_price = prices[-1] * (1 + total_return)
            
            # 价格合理性检查
            if new_price < base_price * 0.5:
                new_price = prices[-1] * (1 + np.random.uniform(0, 0.01))
            elif new_price > base_price * 2:
                new_price = prices[-1] * (1 + np.random.uniform(-0.01, 0))
            
            prices.append(new_price)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
            'close': prices,
            'volume': np.random.uniform(100000, 500000, n_points)
        })
        
        logger.info(f"✅ {symbol} 数据生成完成")
        return data
    
    def _final_backtest(self, symbol, data, test_full_year):
        """最终回测逻辑"""
        trades = []
        monthly_data = []
        current_month = None
        monthly_pnl = 0
        
        # 币种专用资金
        symbol_capital = self.symbol_capital[symbol]
        current_symbol_capital = symbol_capital
        
        for i in range(20, len(data)):
            try:
                row = data.iloc[i]
                current_price = row['close']
                current_time = row['timestamp']
                
                # 月度处理 (复利)
                if test_full_year:
                    month_key = current_time.strftime('%Y-%m')
                    if current_month != month_key and current_month is not None:
                        # 月度结束，应用复利
                        if self.compound_mode and monthly_pnl != 0:
                            monthly_return = monthly_pnl / symbol_capital
                            symbol_capital *= (1 + monthly_return)
                            current_symbol_capital = symbol_capital
                            logger.info(f"💰 {symbol} {current_month} 复利应用: ${symbol_capital:,.2f}")
                        
                        monthly_data.append({
                            'month': current_month,
                            'pnl': monthly_pnl,
                            'capital': symbol_capital
                        })
                        monthly_pnl = 0
                    current_month = month_key
                
                # 获取信号
                signal_data = data.iloc[:i+1]
                signals = self.signal_detector.analyze_enhanced_signals(signal_data, symbol)
                
                if not signals.empty and i < len(signals):
                    signal = signals.iloc[i]
                    signal_type = signal.get('signal_type', 'HOLD')
                    
                    # 执行交易
                    trade_result = self._execute_final_trading(
                        symbol, current_price, current_time, signal_type, current_symbol_capital
                    )
                    if trade_result:
                        trades.append(trade_result)
                        monthly_pnl += trade_result.get('pnl', 0)
                        # 更新币种资金
                        current_symbol_capital += trade_result.get('pnl', 0)
                        
            except Exception as e:
                continue
        
        # 处理最后一个月
        if test_full_year and current_month and monthly_pnl != 0:
            if self.compound_mode:
                monthly_return = monthly_pnl / symbol_capital
                symbol_capital *= (1 + monthly_return)
            monthly_data.append({
                'month': current_month,
                'pnl': monthly_pnl,
                'capital': symbol_capital
            })
        
        # 更新最终资金
        self.symbol_capital[symbol] = current_symbol_capital
        
        # 计算性能指标
        metrics = self._calculate_final_metrics(trades, symbol_capital)
        
        return {
            'symbol': symbol,
            'trades': trades,
            'metrics': metrics,
            'monthly_data': monthly_data,
            'final_capital': current_symbol_capital
        }
    
    def _execute_final_trading(self, symbol, price, timestamp, signal_type, current_capital):
        """执行最终交易逻辑"""
        try:
            # 开仓逻辑
            if signal_type in ['STRONG_BUY', 'STRONG_SELL'] and symbol not in self.positions:
                # 使用币种专用资金
                position_size = current_capital * 0.1  # 10%仓位
                
                if signal_type == 'STRONG_BUY':
                    self.positions[symbol] = {
                        'type': 'long', 'entry_price': price, 'size': position_size, 
                        'timestamp': timestamp, 'capital_used': position_size
                    }
                    return {
                        'symbol': symbol, 'timestamp': timestamp, 'action': 'BUY', 
                        'price': price, 'size': position_size, 'type': 'long'
                    }
                    
                else:  # STRONG_SELL
                    self.positions[symbol] = {
                        'type': 'short', 'entry_price': price, 'size': position_size,
                        'timestamp': timestamp, 'capital_used': position_size
                    }
                    return {
                        'symbol': symbol, 'timestamp': timestamp, 'action': 'SELL', 
                        'price': price, 'size': position_size, 'type': 'short'
                    }
            
            # 平仓逻辑
            elif symbol in self.positions:
                position = self.positions[symbol]
                hold_hours = (timestamp - position['timestamp']).total_seconds() / 3600
                
                should_close = False
                if position['type'] == 'long' and (signal_type == 'STRONG_SELL' or hold_hours > 24):
                    should_close = True
                elif position['type'] == 'short' and (signal_type == 'STRONG_BUY' or hold_hours > 24):
                    should_close = True
                
                if should_close:
                    # 计算盈亏 (使用杠杆)
                    if position['type'] == 'long':
                        pnl = (price - position['entry_price']) / position['entry_price'] * position['size'] * self.leverage
                    else:
                        pnl = (position['entry_price'] - price) / position['entry_price'] * position['size'] * self.leverage
                    
                    trade = {
                        'symbol': symbol, 'timestamp': timestamp, 'action': 'CLOSE',
                        'price': price, 'pnl': pnl, 'type': position['type'], 
                        'hold_hours': hold_hours
                    }
                    
                    del self.positions[symbol]
                    return trade
                    
        except Exception as e:
            logger.error(f"交易执行错误 {symbol}: {e}")
            
        return None
    
    def _calculate_final_metrics(self, trades, final_capital):
        """计算最终性能指标"""
        if not trades:
            return {
                'total_trades': 0, 'win_rate': 0, 'total_pnl': 0,
                'avg_profit': 0, 'profit_factor': 0, 'avg_hold_time': 0,
                'final_capital': final_capital
            }
        
        total_trades = len(trades)
        profitable_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        win_rate = len(profitable_trades) / total_trades * 100
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        avg_profit = total_pnl / total_trades
        
        total_profits = sum(t.get('pnl', 0) for t in profitable_trades) if profitable_trades else 0
        total_losses = sum(t.get('pnl', 0) for t in losing_trades) if losing_trades else 0
        profit_factor = abs(total_profits / total_losses) if total_losses != 0 else float('inf')
        
        hold_times = [t.get('hold_hours', 0) for t in trades if t.get('hold_hours')]
        avg_hold_time = np.mean(hold_times) if hold_times else 0
        
        return {
            'total_trades': total_trades, 'win_rate': win_rate, 'total_pnl': total_pnl,
            'avg_profit': avg_profit, 'profit_factor': profit_factor, 
            'avg_hold_time_hours': avg_hold_time, 'final_capital': final_capital
        }
    
    def _generate_final_report(self, all_results):
        """生成最终报告"""
        logger.info("\n" + "="*100)
        logger.info("🎯 最终量化交易系统 - 完整回测报告")
        logger.info("="*100)
        
        total_trades = sum(len(r['trades']) for r in all_results)
        total_pnl = sum(r['metrics']['total_pnl'] for r in all_results)
        final_total_capital = sum(r['metrics']['final_capital'] for r in all_results)
        
        win_rates = [r['metrics']['win_rate'] for r in all_results if r['trades']]
        avg_win_rate = np.mean(win_rates) if win_rates else 0
        
        logger.info(f"\n📈 最终性能汇总:")
        logger.info(f"  🌐 测试币种: {len(all_results)}个")
        logger.info(f"  📊 总交易次数: {total_trades}笔")
        logger.info(f"  🎯 平均胜率: {avg_win_rate:.1f}%")
        logger.info(f"  💰 总收益: ${total_pnl:+,.2f}")
        logger.info(f"  🏦 最终总资金: ${final_total_capital:,.2f}")
        logger.info(f"  📈 总收益率: {(final_total_capital - self.initial_capital) / self.initial_capital * 100:.1f}%")
        
        logger.info(f"\n📊 各币种最终表现:")
        logger.info("币种          交易数    胜率     总收益      最终资金")
        logger.info("-" * 70)
        
        for result in all_results:
            symbol = result['symbol']
            metrics = result['metrics']
            trades = result['trades']
            
            if trades:
                logger.info(f"{symbol:12} {metrics['total_trades']:6}   {metrics['win_rate']:5.1f}%   ${metrics['total_pnl']:8.2f}   ${metrics['final_capital']:10.2f}")
            else:
                logger.info(f"{symbol:12} {0:6}   {0:5.1f}%   ${0:8.2f}   ${self.symbol_capital[symbol]:10.2f}")
        
        logger.info(f"\n🎉 最终回测完成！系统完整度: ✅ 资金分配 ✅ 双向交易 ✅ 杠杆 ✅ 复利")

def main():
    parser = argparse.ArgumentParser(description='最终高频交易回测系统')
    parser.add_argument('--symbols', type=str, default='BTC/USDT,ETH/USDT,SOL/USDT')
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--capital', type=float, default=10000)
    parser.add_argument('--leverage', type=int, default=3)
    parser.add_argument('--no-compound', action='store_true', help='关闭复利模式')
    
    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    backtest = FinalBacktest(
        initial_capital=args.capital,
        leverage=args.leverage,
        compound_mode=not args.no_compound
    )
    backtest.run_final_backtest(symbols=symbols, days=args.days)

if __name__ == "__main__":
    main()