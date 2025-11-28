#!/usr/bin/env python3
"""
高频交易回测系统 - 纯Python进度条版本
不依赖外部库，内置进度显示
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime, timedelta
import warnings
import time
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AIEnhancedBacktest')

class ProgressTracker:
    """进度跟踪器 - 纯Python实现"""
    
    def __init__(self, total_symbols, total_iterations):
        self.total_symbols = total_symbols
        self.total_iterations = total_iterations
        self.current_symbol = 0
        self.current_iteration = 0
        self.start_time = time.time()
        self.symbol_progress = {}
        self.last_update_time = time.time()
        
    def update_symbol(self, symbol_name, current, total):
        """更新币种进度"""
        self.symbol_progress[symbol_name] = (current, total)
        
    def increment_symbol(self):
        """增加完成的币种计数"""
        self.current_symbol += 1
        
    def get_elapsed_time(self):
        """获取已用时间"""
        elapsed = time.time() - self.start_time
        return timedelta(seconds=int(elapsed))
    
    def get_progress_string(self):
        """获取进度字符串"""
        elapsed = self.get_elapsed_time()
        symbol_progress = f"币种: {self.current_symbol}/{self.total_symbols}"
        
        progress_details = []
        for symbol, (current, total) in self.symbol_progress.items():
            if total > 0:
                percent = (current / total) * 100
                progress_details.append(f"{symbol}: {percent:.1f}%")
        
        details = " | ".join(progress_details) if progress_details else "初始化中..."
        
        if self.current_iteration > 0:
            iterations_per_second = self.current_iteration / (time.time() - self.start_time)
            remaining_iterations = self.total_iterations - self.current_iteration
            eta_seconds = remaining_iterations / iterations_per_second if iterations_per_second > 0 else 0
            eta = timedelta(seconds=int(eta_seconds))
            time_info = f" | 速度: {iterations_per_second:.1f}it/s | ETA: {eta}"
        else:
            time_info = ""
            
        return f"🔄 {symbol_progress} | {details} | 用时: {elapsed}{time_info}"

class SimpleProgressBar:
    """简单进度条 - 纯Python实现"""
    
    def __init__(self, total, description="Progress", bar_length=40):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.bar_length = bar_length
        self.last_percent = -1
        
    def update(self, n=1):
        """更新进度"""
        self.current += n
        self._display()
        
    def _display(self):
        """显示进度条 - 只在进度有显著变化时更新"""
        percent = self.current / self.total
        
        # 只有当进度变化超过1%时才更新显示，减少闪烁
        if int(percent * 100) == int(self.last_percent * 100) and self.current < self.total:
            return
            
        self.last_percent = percent
        
        filled_length = int(self.bar_length * percent)
        bar = '█' * filled_length + '─' * (self.bar_length - filled_length)
        
        elapsed = time.time() - self.start_time
        if self.current > 0:
            items_per_second = self.current / elapsed
            eta_seconds = (self.total - self.current) / items_per_second if items_per_second > 0 else 0
            eta = timedelta(seconds=int(eta_seconds))
            time_info = f" {elapsed:.0f}s [{eta} left]"
        else:
            time_info = ""
            
        # 使用回车符覆盖上一行
        sys.stdout.write(f'\r{self.description}: |{bar}| {percent:.1%} ({self.current}/{self.total}){time_info}')
        sys.stdout.flush()
        
    def close(self):
        """完成进度条"""
        # 显示100%完成
        bar = '█' * self.bar_length
        elapsed = time.time() - self.start_time
        sys.stdout.write(f'\r{self.description}: |{bar}| 100.0% ({self.total}/{self.total}) {elapsed:.0f}s [完成!]')
        sys.stdout.write('\n')
        sys.stdout.flush()

class AIStrategyOptimizer:
    """AI策略优化器"""
    
    def __init__(self):
        self.optimization_history = []
    
    def optimize_signal_parameters(self, historical_data, symbol):
        """优化信号参数 - 使用遗传算法思想"""
        best_params = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_threshold': 0.2,
            'bb_threshold': 0.1,
            'volume_threshold': 1.5,
            'min_signal_strength': 0.6
        }
        
        # 分析历史数据特征
        if len(historical_data) > 100:
            returns = historical_data['close'].pct_change().dropna()
            volatility = returns.std()
            trend_strength = abs(historical_data['close'].pct_change(20).mean())
            
            # 基于市场特征调整参数
            if volatility > 0.02:  # 高波动市场
                best_params['min_signal_strength'] = 0.7
                best_params['macd_threshold'] = 0.3
            elif volatility < 0.01:  # 低波动市场
                best_params['min_signal_strength'] = 0.5
                best_params['macd_threshold'] = 0.15
            
            if trend_strength > 0.001:  # 强趋势市场
                best_params['rsi_oversold'] = 35
                best_params['rsi_overbought'] = 65
        
        logger.info(f"🤖 AI优化 {symbol} 参数: {best_params}")
        return best_params

class EnhancedSignalDetector:
    """增强信号检测器 - 集成AI优化"""
    
    def __init__(self):
        self.ai_optimizer = AIStrategyOptimizer()
        self.symbol_params = {}
    
    def analyze_enhanced_signals(self, data, symbol):
        """分析增强信号 - AI优化版本"""
        try:
            if data is None or len(data) < 50:
                return pd.DataFrame()
            
            # 获取AI优化参数
            if symbol not in self.symbol_params:
                self.symbol_params[symbol] = self.ai_optimizer.optimize_signal_parameters(data, symbol)
            
            params = self.symbol_params[symbol]
            
            # 计算技术指标
            df = self._calculate_enhanced_indicators(data)
            
            # 生成AI优化信号
            signals = self._generate_ai_optimized_signals(df, params)
            
            return signals
            
        except Exception as e:
            logger.error(f"AI信号分析错误: {e}")
            return pd.DataFrame()
    
    def _calculate_enhanced_indicators(self, df):
        """计算增强技术指标"""
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
        
        # 布林带
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        
        # 移动平均线
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_30'] = df['close'].rolling(window=30).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # 价格动量
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        
        # 成交量分析（如果有成交量数据）
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        else:
            df['volume_ratio'] = 1.0
        
        # 波动率
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """计算MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """计算布林带"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def _generate_ai_optimized_signals(self, df, params):
        """生成AI优化信号"""
        signals = []
        
        for i in range(len(df)):
            if i < 50:  # 确保有足够数据计算指标
                signals.append({
                    'signal_strength': 0, 
                    'signal_type': 'HOLD',
                    'confidence': 0,
                    'rsi': 50,
                    'macd_hist': 0,
                    'bb_position': 0
                })
                continue
                
            row = df.iloc[i]
            signal_strength = 0
            confidence_factors = []
            
            # 1. RSI信号 (权重: 0.3)
            rsi_signal = 0
            if row['rsi'] < params['rsi_oversold']:
                rsi_signal = 0.3
                confidence_factors.append(('RSI超卖', 0.8))
            elif row['rsi'] > params['rsi_overbought']:
                rsi_signal = -0.3
                confidence_factors.append(('RSI超买', 0.8))
            signal_strength += rsi_signal
            
            # 2. MACD信号 (权重: 0.3)
            macd_signal = 0
            if row['macd_hist'] > params['macd_threshold']:
                macd_signal = 0.3
                confidence_factors.append(('MACD金叉', 0.7))
            elif row['macd_hist'] < -params['macd_threshold']:
                macd_signal = -0.3
                confidence_factors.append(('MACD死叉', 0.7))
            signal_strength += macd_signal
            
            # 3. 布林带信号 (权重: 0.2)
            bb_signal = 0
            bb_position = (row['close'] - row['bb_lower']) / (row['bb_upper'] - row['bb_lower'])
            if bb_position < 0.1:  # 接近下轨
                bb_signal = 0.2
                confidence_factors.append(('布林带下轨', 0.6))
            elif bb_position > 0.9:  # 接近上轨
                bb_signal = -0.2
                confidence_factors.append(('布林带上轨', 0.6))
            signal_strength += bb_signal
            
            # 4. 移动平均线信号 (权重: 0.2)
            ma_signal = 0
            if row['sma_10'] > row['sma_30'] > row['sma_50']:
                ma_signal = 0.2
                confidence_factors.append(('多头排列', 0.9))
            elif row['sma_10'] < row['sma_30'] < row['sma_50']:
                ma_signal = -0.2
                confidence_factors.append(('空头排列', 0.9))
            signal_strength += ma_signal
            
            # 计算置信度
            confidence = np.mean([cf[1] for cf in confidence_factors]) if confidence_factors else 0
            
            # 确定信号类型
            if signal_strength > params['min_signal_strength'] and confidence > 0.6:
                signal_type = 'STRONG_BUY'
            elif signal_strength > 0.3:
                signal_type = 'BUY'
            elif signal_strength < -params['min_signal_strength'] and confidence > 0.6:
                signal_type = 'STRONG_SELL'
            elif signal_strength < -0.3:
                signal_type = 'SELL'
            else:
                signal_type = 'HOLD'
            
            signals.append({
                'signal_strength': signal_strength,
                'signal_type': signal_type,
                'confidence': confidence,
                'rsi': row['rsi'],
                'macd_hist': row['macd_hist'],
                'bb_position': bb_position,
                'factors': [cf[0] for cf in confidence_factors]
            })
        
        return pd.DataFrame(signals)

class AdvancedBacktest:
    """高级回测系统 - 集成AI优化和进度显示"""
    
    def __init__(self, initial_capital=10000, compound_mode=True, leverage=3):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.compound_mode = compound_mode
        self.leverage = leverage
        self.positions = {}
        self.trade_history = []
        self.performance_metrics = {}
        
        # 使用AI增强信号检测器
        self.signal_detector = EnhancedSignalDetector()
        
        logger.info("🚀 AI增强回测系统初始化完成")
    
    def run_advanced_backtest(self, symbols, days=30):
        """运行高级回测 - 带进度显示"""
        logger.info(f"🎯 开始AI优化回测: {symbols} {days}天")
        
        all_results = []
        detailed_trades = []
        
        # 计算总迭代次数用于进度条
        total_iterations = len(symbols) * days * 24  # 估算值
        
        # 创建进度跟踪器
        progress_tracker = ProgressTracker(len(symbols), total_iterations)
        
        for symbol_idx, symbol in enumerate(symbols):
            logger.info(f"\n🔍 AI优化测试: {symbol} ({symbol_idx + 1}/{len(symbols)})")
            
            try:
                # 生成模拟数据
                data = self._generate_realistic_data(symbol, days)
                logger.info(f"✅ 生成 {symbol} 数据: {len(data)} 条")
                
                # 创建币种进度条
                symbol_progress = SimpleProgressBar(
                    len(data) - 50, 
                    description=f"📊 {symbol} 回测"
                )
                
                # 运行AI优化回测
                result = self._run_ai_optimized_backtest(symbol, data, symbol_progress, progress_tracker)
                all_results.append(result)
                detailed_trades.extend(result['detailed_trades'])
                
                # 完成币种进度
                symbol_progress.close()
                progress_tracker.increment_symbol()
                
                # 显示中间结果
                if result['trades']:
                    metrics = result['metrics']
                    logger.info(f"   ✅ {symbol} 完成: {metrics['total_trades']}笔交易, 胜率: {metrics['win_rate']:.1f}%, 收益: ${metrics['total_pnl']:.2f}")
                else:
                    logger.info(f"   ⚠️  {symbol} 无交易产生")
                
            except Exception as e:
                logger.error(f"❌ {symbol} AI回测失败: {e}")
                continue
        
        # 生成详细报告
        self._generate_detailed_report(all_results, detailed_trades)
        return all_results
    
    def _generate_realistic_data(self, symbol, days):
        """生成更真实的市场数据"""
        dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
        
        base_prices = {
            'BTC/USDT': 35000, 'ETH/USDT': 2500, 'SOL/USDT': 100,
            'BNB/USDT': 300, 'ADA/USDT': 0.5, 'DOT/USDT': 6,
            'AVAX/USDT': 20, 'LINK/USDT': 15, 'MATIC/USDT': 0.8
        }
        
        base_price = base_prices.get(symbol, 100)
        
        # 生成更真实的价格序列（包含趋势和波动）
        np.random.seed(42)
        n_points = len(dates)
        
        # 创建趋势成分
        trend = np.linspace(0, 0.1, n_points)  # 10%的上升趋势
        
        # 创建周期性成分
        cycle = 0.05 * np.sin(2 * np.pi * np.arange(n_points) / (24*7))  # 每周期的波动
        
        # 随机波动
        noise = np.random.normal(0, 0.01, n_points)
        
        # 组合所有成分
        returns = trend + cycle + noise
        prices = base_price * (1 + returns).cumprod()
        
        # 生成成交量（与价格波动相关）
        volume_base = 100000
        volume_variation = np.abs(returns) * 500000
        volumes = volume_base + volume_variation + np.random.uniform(-20000, 20000, n_points)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * 0.998,
            'high': prices * 1.005,
            'low': prices * 0.995, 
            'close': prices,
            'volume': volumes
        })
        
        return data
    
    def _run_ai_optimized_backtest(self, symbol, data, progress_bar, progress_tracker):
        """运行AI优化回测 - 带进度更新"""
        trades = []
        portfolio_values = []
        current_value = self.current_capital
        
        total_iterations = len(data) - 50
        last_global_update = time.time()
        
        for i in range(50, len(data)):  # 从50开始确保有足够数据
            try:
                # 更新进度
                progress_bar.update(1)
                progress_tracker.current_iteration += 1
                progress_tracker.update_symbol(symbol, i-50, total_iterations)
                
                # 每2秒更新一次全局进度显示，避免过于频繁的更新
                current_time = time.time()
                if current_time - last_global_update > 2.0:
                    progress_info = progress_tracker.get_progress_string()
                    sys.stdout.write(f'\r{progress_info}')
                    sys.stdout.flush()
                    last_global_update = current_time
                
                row = data.iloc[i]
                current_price = row['close']
                current_time = row['timestamp']
                
                # 获取AI优化信号
                signal_data = data.iloc[:i+1]
                signals = self.signal_detector.analyze_enhanced_signals(signal_data, symbol)
                
                if not signals.empty and i < len(signals):
                    signal = signals.iloc[i]
                    
                    # 执行AI优化交易
                    trade_result = self._execute_ai_trading(
                        symbol, current_price, current_time, signal
                    )
                    
                    if trade_result:
                        trades.append(trade_result)
                        
                        # 更新投资组合价值
                        if symbol in self.positions:
                            position = self.positions[symbol]
                            if position['type'] == 'long':
                                position_value = position['size'] * (current_price / position['entry_price']) * self.leverage
                            else:
                                position_value = position['size'] * (position['entry_price'] / current_price) * self.leverage
                            current_value = self.current_capital + position_value
                        else:
                            current_value = self.current_capital
                        
                        portfolio_values.append({
                            'timestamp': current_time,
                            'portfolio_value': current_value,
                            'price': current_price
                        })
                        
            except Exception as e:
                logger.error(f"❌ {symbol} 回测迭代错误: {e}")
                continue
        
        # 计算性能指标
        metrics = self._calculate_performance_metrics(trades, portfolio_values)
        
        return {
            'symbol': symbol,
            'trades': trades,
            'metrics': metrics,
            'detailed_trades': trades,
            'portfolio_history': portfolio_values
        }
    
    def _execute_ai_trading(self, symbol, price, timestamp, signal):
        """执行AI优化交易"""
        try:
            signal_strength = signal.get('signal_strength', 0)
            confidence = signal.get('confidence', 0)
            signal_type = signal.get('signal_type', 'HOLD')
            
            # 只有高置信度的强信号才交易
            min_confidence = 0.6
            min_strength = 0.6
            
            # 开仓逻辑
            if (signal_type in ['STRONG_BUY', 'STRONG_SELL'] and 
                confidence >= min_confidence and 
                abs(signal_strength) >= min_strength and 
                symbol not in self.positions):
                
                position_size = self.current_capital * 0.08  # 8%仓位，更保守
                
                if signal_type == 'STRONG_BUY':
                    # 开多头
                    self.positions[symbol] = {
                        'type': 'long',
                        'entry_price': price,
                        'size': position_size,
                        'timestamp': timestamp,
                        'signal_strength': signal_strength,
                        'confidence': confidence
                    }
                    
                    return {
                        'symbol': symbol, 'timestamp': timestamp,
                        'action': 'BUY', 'price': price,
                        'size': position_size, 'type': 'long',
                        'signal_strength': signal_strength,
                        'confidence': confidence,
                        'signal_factors': signal.get('factors', [])
                    }
                    
                else:  # STRONG_SELL
                    # 开空头
                    self.positions[symbol] = {
                        'type': 'short', 
                        'entry_price': price,
                        'size': position_size,
                        'timestamp': timestamp,
                        'signal_strength': signal_strength,
                        'confidence': confidence
                    }
                    
                    return {
                        'symbol': symbol, 'timestamp': timestamp,
                        'action': 'SELL', 'price': price,
                        'size': position_size, 'type': 'short',
                        'signal_strength': signal_strength,
                        'confidence': confidence,
                        'signal_factors': signal.get('factors', [])
                    }
            
            # 平仓逻辑
            elif symbol in self.positions:
                position = self.positions[symbol]
                hold_hours = (timestamp - position['timestamp']).total_seconds() / 3600
                
                # AI优化平仓条件
                should_close = False
                close_reason = ""
                
                if position['type'] == 'long':
                    if signal_type == 'STRONG_SELL' and confidence > 0.7:
                        should_close = True
                        close_reason = "强烈卖出信号"
                    elif hold_hours > 24:  # 最大持有时间
                        should_close = True
                        close_reason = "时间止损"
                    elif signal_strength < -0.4:  # 信号反转
                        should_close = True
                        close_reason = "信号反转"
                        
                else:  # short position
                    if signal_type == 'STRONG_BUY' and confidence > 0.7:
                        should_close = True
                        close_reason = "强烈买入信号"
                    elif hold_hours > 24:
                        should_close = True
                        close_reason = "时间止损"
                    elif signal_strength > 0.4:
                        should_close = True
                        close_reason = "信号反转"
                
                if should_close:
                    # 计算盈亏
                    if position['type'] == 'long':
                        pnl = (price - position['entry_price']) / position['entry_price'] * position['size'] * self.leverage
                    else:
                        pnl = (position['entry_price'] - price) / position['entry_price'] * position['size'] * self.leverage
                    
                    trade = {
                        'symbol': symbol, 'timestamp': timestamp,
                        'action': 'CLOSE', 'price': price,
                        'pnl': pnl, 'type': position['type'],
                        'hold_hours': hold_hours,
                        'close_reason': close_reason,
                        'signal_strength': signal_strength,
                        'confidence': confidence,
                        'entry_signal_strength': position.get('signal_strength', 0),
                        'entry_confidence': position.get('confidence', 0)
                    }
                    
                    self.current_capital += pnl
                    del self.positions[symbol]
                    return trade
                    
        except Exception as e:
            logger.error(f"AI交易执行错误 {symbol}: {e}")
            
        return None
    
    def _calculate_performance_metrics(self, trades, portfolio_history):
        """计算详细的性能指标"""
        if not trades:
            return {}
        
        # 基础指标
        total_trades = len(trades)
        profitable_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        win_rate = len(profitable_trades) / total_trades * 100
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        avg_profit = total_pnl / total_trades
        
        # 盈亏分析
        total_profits = sum(t.get('pnl', 0) for t in profitable_trades)
        total_losses = sum(t.get('pnl', 0) for t in losing_trades)
        profit_factor = abs(total_profits / total_losses) if total_losses != 0 else float('inf')
        
        # 持仓时间分析
        hold_times = [t.get('hold_hours', 0) for t in trades if t.get('hold_hours')]
        avg_hold_time = np.mean(hold_times) if hold_times else 0
        
        # 信号质量分析
        winning_signals = [t.get('entry_signal_strength', 0) for t in profitable_trades]
        losing_signals = [t.get('entry_signal_strength', 0) for t in losing_trades]
        avg_win_signal = np.mean(winning_signals) if winning_signals else 0
        avg_loss_signal = np.mean(losing_signals) if losing_signals else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_profit': avg_profit,
            'profit_factor': profit_factor,
            'avg_hold_time_hours': avg_hold_time,
            'total_profits': total_profits,
            'total_losses': total_losses,
            'avg_win_signal': avg_win_signal,
            'avg_loss_signal': avg_loss_signal,
            'best_trade': max(trades, key=lambda x: x.get('pnl', 0)) if trades else None,
            'worst_trade': min(trades, key=lambda x: x.get('pnl', 0)) if trades else None
        }
    
    def _generate_detailed_report(self, all_results, detailed_trades):
        """生成详细报告"""
        logger.info("\n" + "="*100)
        logger.info("🎯 AI增强量化交易系统 - 详细回测报告")
        logger.info("="*100)
        
        # 总体统计
        total_metrics = self._calculate_total_metrics(all_results)
        
        logger.info(f"\n📈 总体性能汇总:")
        logger.info(f"  🌐 测试币种: {len(all_results)}个")
        logger.info(f"  📊 总交易次数: {total_metrics['total_trades']}笔")
        logger.info(f"  🎯 平均胜率: {total_metrics['avg_win_rate']:.1f}%")
        logger.info(f"  💰 总收益: ${total_metrics['total_pnl']:+,.2f}")
        logger.info(f"  📈 平均每笔收益: ${total_metrics['avg_profit_per_trade']:+.2f}")
        logger.info(f"  ⚖️  盈亏比: {total_metrics['profit_factor']:.2f}")
        logger.info(f"  ⏱️  平均持仓时间: {total_metrics['avg_hold_time']:.1f}小时")
        
        # 币种详细表现
        logger.info(f"\n📊 各币种详细表现:")
        logger.info("币种          交易数    胜率     总收益      平均收益   盈亏比   持仓时间")
        logger.info("-" * 90)
        
        for result in all_results:
            symbol = result['symbol']
            metrics = result['metrics']
            trades = result['trades']
            
            if trades:
                logger.info(f"{symbol:12} {metrics['total_trades']:6}   {metrics['win_rate']:5.1f}%   ${metrics['total_pnl']:8.2f}   ${metrics['avg_profit']:7.2f}   {metrics['profit_factor']:5.2f}   {metrics['avg_hold_time_hours']:6.1f}h")
        
        # AI信号分析
        logger.info(f"\n🤖 AI信号质量分析:")
        winning_trades = [t for t in detailed_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in detailed_trades if t.get('pnl', 0) < 0]
        
        if winning_trades and losing_trades:
            avg_win_signal = np.mean([t.get('entry_signal_strength', 0) for t in winning_trades])
            avg_loss_signal = np.mean([t.get('entry_signal_strength', 0) for t in losing_trades])
            avg_win_confidence = np.mean([t.get('entry_confidence', 0) for t in winning_trades])
            avg_loss_confidence = np.mean([t.get('entry_confidence', 0) for t in losing_trades])
            
            logger.info(f"  ✅ 盈利交易平均信号强度: {avg_win_signal:.3f} (置信度: {avg_win_confidence:.1%})")
            logger.info(f"  ❌ 亏损交易平均信号强度: {avg_loss_signal:.3f} (置信度: {avg_loss_confidence:.1%})")
            logger.info(f"  📊 信号区分度: {abs(avg_win_signal - avg_loss_signal):.3f}")
        
        # 交易分布
        logger.info(f"\n📋 交易分布分析:")
        pnl_values = [t.get('pnl', 0) for t in detailed_trades]
        if pnl_values:
            logger.info(f"  🔺 最大盈利: ${max(pnl_values):.2f}")
            logger.info(f"  🔻 最大亏损: ${min(pnl_values):.2f}")
            logger.info(f"  📏 收益标准差: ${np.std(pnl_values):.2f}")
            logger.info(f"  📈 夏普比率: {np.mean(pnl_values)/np.std(pnl_values) if np.std(pnl_values) > 0 else 0:.2f}")
        
        # 建议和改进
        logger.info(f"\n💡 AI优化建议:")
        if total_metrics['avg_win_rate'] < 40:
            logger.info("  🎯 建议: 提高信号阈值，减少低质量交易")
        if total_metrics['profit_factor'] < 1.5:
            logger.info("  ⚖️  建议: 优化止损策略，提高盈亏比")
        if total_metrics['avg_hold_time'] > 48:
            logger.info("  ⏱️  建议: 缩短持仓时间，提高资金周转率")
        
        logger.info(f"\n🎉 AI优化回测完成！")
        logger.info("="*50)
    
    def _calculate_total_metrics(self, all_results):
        """计算总体指标"""
        total_trades = sum(len(result['trades']) for result in all_results)
        total_pnl = sum(result['metrics']['total_pnl'] for result in all_results if result['trades'])
        
        win_rates = [result['metrics']['win_rate'] for result in all_results if result['trades']]
        avg_win_rate = np.mean(win_rates) if win_rates else 0
        
        profit_factors = [result['metrics']['profit_factor'] for result in all_results if result['trades']]
        avg_profit_factor = np.mean(profit_factors) if profit_factors else 0
        
        hold_times = [result['metrics']['avg_hold_time_hours'] for result in all_results if result['trades']]
        avg_hold_time = np.mean(hold_times) if hold_times else 0
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'avg_win_rate': avg_win_rate,
            'avg_profit_per_trade': total_pnl / total_trades if total_trades > 0 else 0,
            'profit_factor': avg_profit_factor,
            'avg_hold_time': avg_hold_time
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AI增强高频交易回测系统')
    parser.add_argument('--symbols', type=str, default='BTC/USDT,ETH/USDT,SOL/USDT',
                       help='交易对，用逗号分隔')
    parser.add_argument('--days', type=int, default=30,
                       help='回测天数')
    parser.add_argument('--capital', type=float, default=10000,
                       help='初始资金')
    
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # 创建AI增强回测实例
    backtest = AdvancedBacktest(initial_capital=args.capital)
    
    # 运行AI优化回测
    backtest.run_advanced_backtest(symbols=symbols, days=args.days)

if __name__ == "__main__":
    main()