#!/usr/bin/env python3
"""
高频交易回测系统 - 调试版本
找出为什么没有交易产生
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
logger = logging.getLogger('DebugBacktest')

class DebugSignalDetector:
    """调试信号检测器 - 输出详细信号信息"""
    
    def __init__(self):
        self.signal_count = 0
        self.strong_signals = 0
        logger.info("🔍 调试信号检测器初始化完成")
    
    def analyze_enhanced_signals(self, data, symbol):
        """分析增强信号 - 调试版本"""
        try:
            if data is None or len(data) < 50:
                return pd.DataFrame()
            
            # 计算技术指标
            df = self._calculate_technical_indicators(data)
            
            # 生成信号并调试
            signals = self._generate_signals_with_debug(df, symbol)
            
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
        df['sma_30'] = df['close'].rolling(window=30, min_periods=1).mean()
        
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
    
    def _generate_signals_with_debug(self, df, symbol):
        """生成信号 - 带调试信息"""
        signals = []
        
        for i in range(len(df)):
            if i < 50:
                signals.append({
                    'signal_strength': 0, 
                    'signal_type': 'HOLD',
                    'confidence': 0
                })
                continue
                
            try:
                row = df.iloc[i]
                signal_strength = 0
                confidence = 0.5
                signal_details = []
                
                # RSI信号
                rsi = row.get('rsi', 50)
                if rsi < 30:
                    signal_strength += 0.4
                    confidence += 0.2
                    signal_details.append(f"RSI超卖({rsi:.1f})")
                elif rsi > 70:
                    signal_strength -= 0.4
                    confidence += 0.2
                    signal_details.append(f"RSI超买({rsi:.1f})")
                
                # MACD信号
                macd_hist = row.get('macd_hist', 0)
                if macd_hist > 0.1:
                    signal_strength += 0.3
                    confidence += 0.15
                    signal_details.append(f"MACD金叉({macd_hist:.3f})")
                elif macd_hist < -0.1:
                    signal_strength -= 0.3
                    confidence += 0.15
                    signal_details.append(f"MACD死叉({macd_hist:.3f})")
                
                # 移动平均线信号
                sma_10 = row.get('sma_10', 0)
                sma_30 = row.get('sma_30', 0)
                if sma_10 > sma_30:
                    signal_strength += 0.2
                    confidence += 0.1
                    signal_details.append("均线多头")
                elif sma_10 < sma_30:
                    signal_strength -= 0.2
                    confidence += 0.1
                    signal_details.append("均线空头")
                
                # 限制范围
                signal_strength = max(min(signal_strength, 1.0), -1.0)
                confidence = max(min(confidence, 1.0), 0.0)
                
                # 调试：记录信号统计
                self.signal_count += 1
                
                # 确定信号类型 (降低阈值进行测试)
                if signal_strength > 0.5 and confidence > 0.5:
                    signal_type = 'STRONG_BUY'
                    self.strong_signals += 1
                    if self.strong_signals <= 5:  # 只显示前5个强信号
                        logger.info(f"   🎯 强买入信号: 强度={signal_strength:.2f}, 置信度={confidence:.2f}, 因素={signal_details}")
                elif signal_strength > 0.3:
                    signal_type = 'BUY'
                elif signal_strength < -0.5 and confidence > 0.5:
                    signal_type = 'STRONG_SELL'
                    self.strong_signals += 1
                    if self.strong_signals <= 5:  # 只显示前5个强信号
                        logger.info(f"   🎯 强卖出信号: 强度={signal_strength:.2f}, 置信度={confidence:.2f}, 因素={signal_details}")
                elif signal_strength < -0.3:
                    signal_type = 'SELL'
                else:
                    signal_type = 'HOLD'
                
                signals.append({
                    'signal_strength': signal_strength,
                    'signal_type': signal_type,
                    'confidence': confidence,
                    'details': signal_details
                })
                
            except Exception as e:
                signals.append({
                    'signal_strength': 0, 
                    'signal_type': 'HOLD',
                    'confidence': 0,
                    'details': [f"错误: {e}"]
                })
        
        logger.info(f"   📊 {symbol} 信号统计: 总信号={self.signal_count}, 强信号={self.strong_signals}")
        return pd.DataFrame(signals)

class DebugBacktest:
    """调试回测系统"""
    
    def __init__(self, initial_capital=10000, leverage=3):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.leverage = leverage
        self.positions = {}
        self.trade_history = []
        
        # 使用调试信号检测器
        self.signal_detector = DebugSignalDetector()
        
        logger.info("🚀 调试回测系统初始化完成")
    
    def _generate_realistic_data(self, symbol, days):
        """生成真实市场数据"""
        dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
        
        base_prices = {'BTC/USDT': 35000, 'ETH/USDT': 2500, 'SOL/USDT': 100}
        base_price = base_prices.get(symbol, 100)
        n_points = len(dates)
        
        # 创建明显的趋势和波动
        np.random.seed(42)
        
        # 明显趋势
        trend = np.linspace(0, 0.1, n_points)  # 10%趋势
        
        # 周期性波动
        cycle = 0.05 * np.sin(2 * np.pi * np.arange(n_points) / (24*7))
        
        # 随机波动
        noise = np.random.normal(0, 0.01, n_points)
        
        returns = trend + cycle + noise
        prices = base_price * (1 + returns).cumprod()
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices * 0.998,
            'high': prices * 1.005,
            'low': prices * 0.995, 
            'close': prices,
            'volume': np.random.uniform(100000, 500000, n_points)
        })
    
    def run_debug_backtest(self, symbols, days=30):
        """运行调试回测"""
        logger.info(f"🎯 开始调试回测: {symbols} {days}天")
        
        all_results = []
        
        for symbol in symbols:
            logger.info(f"\n🔍 调试币种: {symbol}")
            
            try:
                # 生成数据
                data = self._generate_realistic_data(symbol, days)
                logger.info(f"✅ {symbol} 数据生成: {len(data)} 条")
                
                # 运行回测
                result = self._debug_single_symbol(symbol, data)
                all_results.append(result)
                
            except Exception as e:
                logger.error(f"❌ {symbol} 回测失败: {e}")
                continue
        
        # 生成调试报告
        self._generate_debug_report(all_results)
        return all_results
    
    def _debug_single_symbol(self, symbol, data):
        """单币种调试回测"""
        trades = []
        signal_analysis = []
        
        # 测试前100个数据点来调试
        test_points = min(100, len(data) - 50)
        
        for i in range(50, 50 + test_points):
            try:
                row = data.iloc[i]
                current_price = row['close']
                
                # 获取信号
                signal_data = data.iloc[:i+1]
                signals = self.signal_detector.analyze_enhanced_signals(signal_data, symbol)
                
                if not signals.empty and i < len(signals):
                    signal = signals.iloc[i]
                    signal_strength = signal.get('signal_strength', 0)
                    signal_type = signal.get('signal_type', 'HOLD')
                    confidence = signal.get('confidence', 0)
                    
                    # 记录信号分析
                    signal_analysis.append({
                        'index': i,
                        'price': current_price,
                        'signal_strength': signal_strength,
                        'signal_type': signal_type,
                        'confidence': confidence,
                        'details': signal.get('details', [])
                    })
                    
                    # 执行交易 (宽松条件)
                    if signal_type in ['STRONG_BUY', 'STRONG_SELL'] and symbol not in self.positions:
                        logger.info(f"   💰 尝试开仓: {signal_type} at ${current_price:.2f}")
                        
                        position_size = self.current_capital * 0.1
                        
                        if signal_type == 'STRONG_BUY':
                            self.positions[symbol] = {
                                'type': 'long', 'entry_price': current_price, 
                                'size': position_size, 'timestamp': row['timestamp']
                            }
                            trade = {
                                'symbol': symbol, 'timestamp': row['timestamp'], 
                                'action': 'BUY', 'price': current_price,
                                'size': position_size, 'type': 'long'
                            }
                            trades.append(trade)
                            logger.info(f"   ✅ 成功开多头仓位")
                            
                        else:  # STRONG_SELL
                            self.positions[symbol] = {
                                'type': 'short', 'entry_price': current_price,
                                'size': position_size, 'timestamp': row['timestamp']
                            }
                            trade = {
                                'symbol': symbol, 'timestamp': row['timestamp'],
                                'action': 'SELL', 'price': current_price,
                                'size': position_size, 'type': 'short'
                            }
                            trades.append(trade)
                            logger.info(f"   ✅ 成功开空头仓位")
                
                # 如果有仓位，尝试平仓
                if symbol in self.positions:
                    position = self.positions[symbol]
                    hold_hours = (row['timestamp'] - position['timestamp']).total_seconds() / 3600
                    
                    if hold_hours > 4:  # 短期持有就平仓
                        # 计算盈亏
                        if position['type'] == 'long':
                            pnl = (current_price - position['entry_price']) / position['entry_price'] * position['size'] * self.leverage
                        else:
                            pnl = (position['entry_price'] - current_price) / position['entry_price'] * position['size'] * self.leverage
                        
                        trade = {
                            'symbol': symbol, 'timestamp': row['timestamp'],
                            'action': 'CLOSE', 'price': current_price,
                            'pnl': pnl, 'type': position['type'],
                            'hold_hours': hold_hours
                        }
                        
                        self.current_capital += pnl
                        del self.positions[symbol]
                        trades.append(trade)
                        logger.info(f"   🔒 平仓: 收益=${pnl:+.2f}")
                        
            except Exception as e:
                logger.error(f"❌ 调试迭代错误: {e}")
                continue
        
        # 分析信号质量
        if signal_analysis:
            strong_signals = [s for s in signal_analysis if s['signal_type'] in ['STRONG_BUY', 'STRONG_SELL']]
            avg_strength = np.mean([s['signal_strength'] for s in signal_analysis])
            avg_confidence = np.mean([s['confidence'] for s in signal_analysis])
            
            logger.info(f"   📈 信号质量分析:")
            logger.info(f"     总信号数: {len(signal_analysis)}")
            logger.info(f"     强信号数: {len(strong_signals)}")
            logger.info(f"     平均信号强度: {avg_strength:.3f}")
            logger.info(f"     平均置信度: {avg_confidence:.3f}")
        
        return {
            'symbol': symbol,
            'trades': trades,
            'signal_analysis': signal_analysis,
            'total_trades': len(trades)
        }
    
    def _generate_debug_report(self, all_results):
        """生成调试报告"""
        logger.info("\n" + "="*80)
        logger.info("🔍 调试回测报告")
        logger.info("="*80)
        
        total_trades = sum(r['total_trades'] for r in all_results)
        
        logger.info(f"\n📊 交易统计:")
        logger.info(f"  总交易次数: {total_trades}笔")
        
        for result in all_results:
            symbol = result['symbol']
            trades = result['trades']
            
            logger.info(f"  {symbol}: {len(trades)}笔交易")
            
            # 显示前几个交易的详细信息
            for i, trade in enumerate(trades[:3]):
                logger.info(f"    交易{i+1}: {trade['action']} @ ${trade['price']:.2f}")
        
        if total_trades == 0:
            logger.info(f"\n💡 调试建议:")
            logger.info(f"  1. 检查信号生成逻辑")
            logger.info(f"  2. 降低交易阈值")
            logger.info(f"  3. 验证技术指标计算")
            logger.info(f"  4. 检查数据质量")
        
        logger.info(f"\n🎯 调试完成")

def main():
    parser = argparse.ArgumentParser(description='调试版高频交易回测系统')
    parser.add_argument('--symbols', type=str, default='BTC/USDT,ETH/USDT,SOL/USDT')
    parser.add_argument('--days', type=int, default=10)  # 减少天数用于调试
    parser.add_argument('--capital', type=float, default=10000)
    
    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    backtest = DebugBacktest(initial_capital=args.capital)
    backtest.run_debug_backtest(symbols=symbols, days=args.days)

if __name__ == "__main__":
    main()