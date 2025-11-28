#!/usr/bin/env python3
"""
高频交易回测系统 - 修复信号版本
修复信号数量异常问题
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
logger = logging.getLogger('FixedSignalBacktest')

class FixedSignalDetector:
    """修复信号检测器 - 控制信号数量"""
    
    def __init__(self, use_technical=True, use_random=True):
        self.use_technical = use_technical
        self.use_random = use_random
        self.technical_signals = 0
        self.random_signals = 0
        logger.info(f"🔧 修复信号检测器初始化")
    
    def analyze_enhanced_signals(self, data, symbol):
        """分析增强信号 - 修复版本"""
        try:
            if data is None or len(data) < 20:
                return pd.DataFrame()
            
            signals = []
            total_data_points = len(data)
            
            for i in range(total_data_points):
                if i < 20:
                    signals.append({'signal_strength': 0, 'signal_type': 'HOLD', 'source': 'none'})
                    continue
                
                technical_signal = 'HOLD'
                random_signal = 'HOLD'
                
                # 1. 技术信号 - 只在突破时产生
                if self.use_technical:
                    technical_signal = self._generate_technical_signal(data, i)
                    if technical_signal != 'HOLD':
                        self.technical_signals += 1
                
                # 2. 随机信号 - 严格控制频率
                if self.use_random:
                    random_signal = self._generate_fixed_random_signal(i, total_data_points)
                    if random_signal != 'HOLD':
                        self.random_signals += 1
                
                # 信号合并
                final_signal = self._merge_signals(technical_signal, random_signal)
                signal_strength = 0.7 if final_signal in ['STRONG_BUY', 'STRONG_SELL'] else 0.3
                
                signals.append({
                    'signal_strength': signal_strength,
                    'signal_type': final_signal,
                    'source': 'technical' if final_signal == technical_signal else 'random'
                })
            
            logger.info(f"📊 {symbol} 修复信号: 技术={self.technical_signals}, 随机={self.random_signals}")
            return pd.DataFrame(signals)
            
        except Exception as e:
            logger.error(f"信号分析错误: {e}")
            return pd.DataFrame()
    
    def _generate_technical_signal(self, data, i):
        """生成技术信号 - 严格条件"""
        try:
            if i < 10:
                return 'HOLD'
                
            current_price = data['close'].iloc[i]
            sma_10 = data['close'].iloc[i-10:i].mean()
            
            # 更严格的突破条件
            price_change = (current_price - sma_10) / sma_10
            
            if price_change > 0.02:  # 2%以上突破
                return 'STRONG_BUY'
            elif price_change < -0.02:  # -2%以下突破
                return 'STRONG_SELL'
                
            return 'HOLD'
        except:
            return 'HOLD'
    
    def _generate_fixed_random_signal(self, i, total_points):
        """生成固定随机信号 - 严格控制数量"""
        # 每50个数据点才考虑生成一个随机信号
        if i % 50 == 0:
            rand_val = np.random.random()
            # 更高的质量阈值
            if rand_val > 0.8:  # 20%概率强买入
                return 'STRONG_BUY'
            elif rand_val < 0.2:  # 20%概率强卖出
                return 'STRONG_SELL'
        
        return 'HOLD'
    
    def _merge_signals(self, tech_signal, random_signal):
        """合并信号"""
        if tech_signal != 'HOLD':
            return tech_signal
        elif random_signal != 'HOLD':
            return random_signal
        else:
            return 'HOLD'

class FixedSignalBacktest:
    """修复信号回测系统"""
    
    def __init__(self, initial_capital=10000, leverage=3):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.leverage = leverage
        self.positions = {}
        self.trade_history = []
        
        self.signal_detector = FixedSignalDetector(use_technical=True, use_random=True)
        
        logger.info("🚀 修复信号回测系统初始化完成")
    
    def run_fixed_backtest(self, symbols, days=30):
        """运行修复回测"""
        logger.info(f"🎯 开始修复回测: {symbols} {days}天")
        logger.info(f"📅 回测时间: {datetime.now() - timedelta(days=days)} 到 {datetime.now()}")
        
        all_results = []
        
        for symbol in symbols:
            logger.info(f"\n🔍 修复测试: {symbol}")
            
            try:
                data = self._generate_realistic_data(symbol, days)
                result = self._fixed_backtest(symbol, data)
                all_results.append(result)
                
                metrics = result['metrics']
                if metrics['total_trades'] > 0:
                    logger.info(f"   📈 结果: {metrics['total_trades']}笔, 胜率: {metrics['win_rate']:.1f}%, 收益: ${metrics['total_pnl']:+.2f}")
                
            except Exception as e:
                logger.error(f"❌ {symbol} 回测失败: {e}")
                continue
        
        self._generate_fixed_report(all_results)
        return all_results
    
    def _generate_realistic_data(self, symbol, days):
        """生成真实数据"""
        n_points = days * 24
        dates = pd.date_range(end=datetime.now(), periods=n_points, freq='H')
        
        base_prices = {'BTC/USDT': 35000, 'ETH/USDT': 2500, 'SOL/USDT': 100}
        base_price = base_prices.get(symbol, 100)
        
        np.random.seed(42)
        prices = [base_price]
        
        for i in range(1, n_points):
            daily_return = np.random.uniform(-0.015, 0.015)
            trend = 0.0002
            cycle = 0.003 * np.sin(2 * np.pi * i / (24 * 7))
            
            new_price = prices[-1] * (1 + daily_return + trend + cycle)
            prices.append(new_price)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
            'close': prices,
            'volume': np.random.uniform(100000, 500000, n_points)
        })
    
    def _fixed_backtest(self, symbol, data):
        """修复回测逻辑"""
        trades = []
        
        for i in range(20, len(data)):
            try:
                row = data.iloc[i]
                signals = self.signal_detector.analyze_enhanced_signals(data.iloc[:i+1], symbol)
                
                if not signals.empty and i < len(signals):
                    signal = signals.iloc[i]
                    trade_result = self._execute_trading(symbol, row['close'], row['timestamp'], signal['signal_type'])
                    if trade_result:
                        trades.append(trade_result)
                        
            except Exception:
                continue
        
        metrics = self._calculate_metrics(trades)
        return {
            'symbol': symbol,
            'trades': trades,
            'metrics': metrics
        }
    
    def _execute_trading(self, symbol, price, timestamp, signal_type):
        """执行交易"""
        try:
            if signal_type in ['STRONG_BUY', 'STRONG_SELL'] and symbol not in self.positions:
                position_size = self.current_capital * 0.08
                
                if signal_type == 'STRONG_BUY':
                    self.positions[symbol] = {'type': 'long', 'entry_price': price, 'size': position_size, 'timestamp': timestamp}
                    return {'symbol': symbol, 'timestamp': timestamp, 'action': 'BUY', 'price': price, 'size': position_size, 'type': 'long'}
                else:
                    self.positions[symbol] = {'type': 'short', 'entry_price': price, 'size': position_size, 'timestamp': timestamp}
                    return {'symbol': symbol, 'timestamp': timestamp, 'action': 'SELL', 'price': price, 'size': position_size, 'type': 'short'}
            
            elif symbol in self.positions:
                position = self.positions[symbol]
                hold_hours = (timestamp - position['timestamp']).total_seconds() / 3600
                
                if hold_hours > 18:
                    if position['type'] == 'long':
                        pnl = (price - position['entry_price']) / position['entry_price'] * position['size'] * self.leverage
                    else:
                        pnl = (position['entry_price'] - price) / position['entry_price'] * position['size'] * self.leverage
                    
                    trade = {'symbol': symbol, 'timestamp': timestamp, 'action': 'CLOSE', 'price': price, 'pnl': pnl, 'type': position['type']}
                    self.current_capital += pnl
                    del self.positions[symbol]
                    return trade
                    
        except Exception as e:
            logger.error(f"交易错误: {e}")
            
        return None
    
    def _calculate_metrics(self, trades):
        """计算指标"""
        if not trades:
            return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'avg_profit': 0}
        
        total_trades = len(trades)
        profitable_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        win_rate = profitable_trades / total_trades * 100
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        avg_profit = total_pnl / total_trades
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_profit': avg_profit
        }
    
    def _generate_fixed_report(self, all_results):
        """生成修复报告"""
        logger.info("\n" + "="*80)
        logger.info("🔧 修复信号系统 - 回测报告")
        logger.info("="*80)
        
        total_trades = sum(len(r['trades']) for r in all_results)
        total_pnl = sum(r['metrics']['total_pnl'] for r in all_results)
        
        logger.info(f"📊 总体表现:")
        logger.info(f"  币种数量: {len(all_results)}")
        logger.info(f"  总交易次数: {total_trades}笔")
        logger.info(f"  总收益: ${total_pnl:+,.2f}")
        logger.info(f"  最终资金: ${self.current_capital:,.2f}")
        
        if total_trades > 0:
            win_rates = [r['metrics']['win_rate'] for r in all_results if r['trades']]
            avg_win_rate = np.mean(win_rates) if win_rates else 0
            logger.info(f"  平均胜率: {avg_win_rate:.1f}%")
        
        logger.info(f"\n🎉 修复回测完成！")

def main():
    parser = argparse.ArgumentParser(description='修复信号高频交易回测系统')
    parser.add_argument('--symbols', type=str, default='BTC/USDT,ETH/USDT,SOL/USDT')
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--capital', type=float, default=10000)
    
    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    backtest = FixedSignalBacktest(initial_capital=args.capital)
    backtest.run_fixed_backtest(symbols=symbols, days=args.days)

if __name__ == "__main__":
    main()