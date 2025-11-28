#!/usr/bin/env python3
"""
高频交易回测系统 - 深度诊断版本
找出信号生成的根本问题
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
logger = logging.getLogger('DiagnosticBacktest')

class DiagnosticSignalDetector:
    """诊断信号检测器 - 深入检查每个步骤"""
    
    def __init__(self):
        self.diagnostic_data = []
        logger.info("🔧 诊断信号检测器初始化完成")
    
    def analyze_enhanced_signals(self, data, symbol):
        """分析增强信号 - 深度诊断版本"""
        try:
            logger.info(f"🔍 开始诊断 {symbol} 信号生成...")
            
            if data is None:
                logger.error("❌ 数据为空")
                return pd.DataFrame()
                
            if len(data) < 50:
                logger.error(f"❌ 数据不足: {len(data)} 条，需要至少50条")
                return pd.DataFrame()
            
            logger.info(f"📊 数据基本信息: {len(data)} 条记录")
            logger.info(f"📈 价格范围: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
            logger.info(f"📉 价格变化: {data['close'].iloc[0]:.2f} → {data['close'].iloc[-1]:.2f}")
            
            # 计算技术指标并诊断
            df = self._calculate_technical_indicators_with_diagnosis(data, symbol)
            
            # 生成信号
            signals = self._generate_signals_with_detailed_diagnosis(df, symbol)
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ 信号分析错误: {e}")
            import traceback
            logger.error(f"❌ 详细错误: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def _calculate_technical_indicators_with_diagnosis(self, df, symbol):
        """计算技术指标 - 带详细诊断"""
        logger.info(f"📊 开始计算 {symbol} 技术指标...")
        
        try:
            # RSI
            logger.info("🔧 计算RSI...")
            df['rsi'] = self._calculate_rsi(df['close'])
            rsi_stats = df['rsi'].describe()
            logger.info(f"📈 RSI统计: 均值={rsi_stats['mean']:.1f}, 范围=[{rsi_stats['min']:.1f}, {rsi_stats['max']:.1f}]")
            
            # MACD
            logger.info("🔧 计算MACD...")
            df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
            macd_stats = df['macd_hist'].describe()
            logger.info(f"📈 MACD统计: 均值={macd_stats['mean']:.3f}, 范围=[{macd_stats['min']:.3f}, {macd_stats['max']:.3f}]")
            
            # 移动平均线
            logger.info("🔧 计算移动平均线...")
            df['sma_10'] = df['close'].rolling(window=10, min_periods=1).mean()
            df['sma_30'] = df['close'].rolling(window=30, min_periods=1).mean()
            
            # 显示技术指标示例
            sample_idx = [50, 100, 200]
            for idx in sample_idx:
                if idx < len(df):
                    row = df.iloc[idx]
                    logger.info(f"📊 样本{idx}: 价格=${row['close']:.2f}, RSI={row['rsi']:.1f}, MACD={row['macd_hist']:.3f}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ 技术指标计算错误: {e}")
            import traceback
            logger.error(f"❌ 详细错误: {traceback.format_exc()}")
            return df
    
    def _calculate_rsi(self, prices, period=14):
        """计算RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan).fillna(1)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except Exception as e:
            logger.error(f"❌ RSI计算错误: {e}")
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """计算MACD"""
        try:
            ema_fast = prices.ewm(span=fast, min_periods=1).mean()
            ema_slow = prices.ewm(span=slow, min_periods=1).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal, min_periods=1).mean()
            macd_hist = macd - macd_signal
            return macd, macd_signal, macd_hist
        except Exception as e:
            logger.error(f"❌ MACD计算错误: {e}")
            zeros = pd.Series([0] * len(prices), index=prices.index)
            return zeros, zeros, zeros
    
    def _generate_signals_with_detailed_diagnosis(self, df, symbol):
        """生成信号 - 带详细诊断"""
        logger.info(f"🎯 开始生成 {symbol} 交易信号...")
        
        signals = []
        signal_count = 0
        strong_signal_count = 0
        
        # 检查关键数据点
        test_indices = [50, 100, 150, 200, 250]
        
        for idx in test_indices:
            if idx < len(df):
                row = df.iloc[idx]
                logger.info(f"🔍 检查数据点{idx}: 价格=${row['close']:.2f}, RSI={row.get('rsi', 'N/A')}, MACD={row.get('macd_hist', 'N/A'):.3f}")
        
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
                signal_factors = []
                
                # RSI信号诊断
                rsi = row.get('rsi', 50)
                if not pd.isna(rsi):
                    if rsi < 30:
                        signal_strength += 0.4
                        confidence += 0.2
                        signal_factors.append(f"RSI超卖({rsi:.1f})")
                    elif rsi > 70:
                        signal_strength -= 0.4
                        confidence += 0.2
                        signal_factors.append(f"RSI超买({rsi:.1f})")
                
                # MACD信号诊断
                macd_hist = row.get('macd_hist', 0)
                if not pd.isna(macd_hist):
                    if macd_hist > 0.1:
                        signal_strength += 0.3
                        confidence += 0.15
                        signal_factors.append(f"MACD金叉({macd_hist:.3f})")
                    elif macd_hist < -0.1:
                        signal_strength -= 0.3
                        confidence += 0.15
                        signal_factors.append(f"MACD死叉({macd_hist:.3f})")
                
                # 移动平均线信号诊断
                sma_10 = row.get('sma_10', 0)
                sma_30 = row.get('sma_30', 0)
                if not pd.isna(sma_10) and not pd.isna(sma_30):
                    if sma_10 > sma_30:
                        signal_strength += 0.2
                        confidence += 0.1
                        signal_factors.append("均线多头")
                    elif sma_10 < sma_30:
                        signal_strength -= 0.2
                        confidence += 0.1
                        signal_factors.append("均线空头")
                
                # 限制范围
                signal_strength = max(min(signal_strength, 1.0), -1.0)
                confidence = max(min(confidence, 1.0), 0.0)
                
                signal_count += 1
                
                # 确定信号类型 (非常宽松的条件用于测试)
                if signal_strength > 0.3 or signal_strength < -0.3:
                    if signal_strength > 0.3:
                        signal_type = 'STRONG_BUY'
                    else:
                        signal_type = 'STRONG_SELL'
                    strong_signal_count += 1
                    
                    # 记录前几个强信号的详细信息
                    if strong_signal_count <= 3:
                        logger.info(f"   🎯 强信号{i}: {signal_type}, 强度={signal_strength:.2f}, 置信度={confidence:.2f}")
                        logger.info(f"      因素: {signal_factors}")
                        logger.info(f"      价格: ${row['close']:.2f}, RSI: {rsi:.1f}, MACD: {macd_hist:.3f}")
                else:
                    signal_type = 'HOLD'
                
                signals.append({
                    'signal_strength': signal_strength,
                    'signal_type': signal_type,
                    'confidence': confidence,
                    'factors': signal_factors
                })
                
                # 诊断数据记录
                if i % 100 == 0 and i > 50:
                    self.diagnostic_data.append({
                        'index': i,
                        'price': row['close'],
                        'rsi': rsi,
                        'macd_hist': macd_hist,
                        'signal_strength': signal_strength,
                        'signal_type': signal_type,
                        'factors': signal_factors
                    })
                
            except Exception as e:
                logger.error(f"❌ 信号生成错误(位置{i}): {e}")
                signals.append({
                    'signal_strength': 0, 
                    'signal_type': 'HOLD',
                    'confidence': 0,
                    'factors': [f"错误: {e}"]
                })
        
        logger.info(f"📊 {symbol} 信号生成总结:")
        logger.info(f"  总信号数: {signal_count}")
        logger.info(f"  强信号数: {strong_signal_count}")
        logger.info(f"  强信号比例: {strong_signal_count/max(signal_count,1)*100:.1f}%")
        
        return pd.DataFrame(signals)
    
    def get_diagnostic_summary(self):
        """获取诊断摘要"""
        if not self.diagnostic_data:
            return "无诊断数据"
        
        strengths = [d['signal_strength'] for d in self.diagnostic_data]
        avg_strength = np.mean(strengths) if strengths else 0
        
        return f"诊断样本: {len(self.diagnostic_data)}个, 平均信号强度: {avg_strength:.3f}"

class DiagnosticBacktest:
    """诊断回测系统"""
    
    def __init__(self, initial_capital=10000, leverage=3):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.leverage = leverage
        self.positions = {}
        self.trade_history = []
        
        # 使用诊断信号检测器
        self.signal_detector = DiagnosticSignalDetector()
        
        logger.info("🚀 诊断回测系统初始化完成")
    
    def _generate_realistic_data(self, symbol, days):
        """生成真实市场数据"""
        logger.info(f"📊 生成 {symbol} 模拟数据...")
        
        dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
        
        base_prices = {'BTC/USDT': 35000, 'ETH/USDT': 2500, 'SOL/USDT': 100}
        base_price = base_prices.get(symbol, 100)
        n_points = len(dates)
        
        # 创建包含明显波动的数据
        np.random.seed(42)
        
        # 明显趋势
        trend = np.linspace(0, 0.08, n_points)
        
        # 强周期性波动
        cycle = 0.06 * np.sin(2 * np.pi * np.arange(n_points) / (24*5))
        
        # 随机波动
        noise = np.random.normal(0, 0.015, n_points)
        
        returns = trend + cycle + noise
        prices = base_price * (1 + returns).cumprod()
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * 0.998,
            'high': prices * 1.008,
            'low': prices * 0.992, 
            'close': prices,
            'volume': np.random.uniform(100000, 500000, n_points)
        })
        
        logger.info(f"✅ {symbol} 数据生成完成: {len(data)} 条记录")
        logger.info(f"📈 价格统计: 开=${data['close'].iloc[0]:.2f}, 收=${data['close'].iloc[-1]:.2f}")
        logger.info(f"📊 价格范围: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        
        return data
    
    def run_diagnostic_backtest(self, symbols, days=10):
        """运行诊断回测"""
        logger.info(f"🎯 开始诊断回测: {symbols} {days}天")
        
        all_results = []
        
        for symbol in symbols:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 深度诊断币种: {symbol}")
            logger.info(f"{'='*60}")
            
            try:
                # 生成数据
                data = self._generate_realistic_data(symbol, days)
                
                # 运行诊断
                result = self._diagnose_single_symbol(symbol, data)
                all_results.append(result)
                
            except Exception as e:
                logger.error(f"❌ {symbol} 诊断失败: {e}")
                import traceback
                logger.error(f"❌ 详细错误: {traceback.format_exc()}")
                continue
        
        # 生成诊断报告
        self._generate_diagnostic_report(all_results)
        return all_results
    
    def _diagnose_single_symbol(self, symbol, data):
        """单币种深度诊断"""
        logger.info(f"🔧 开始深度诊断 {symbol}...")
        
        # 获取信号
        signals = self.signal_detector.analyze_enhanced_signals(data, symbol)
        
        # 分析信号质量
        signal_analysis = self._analyze_signals(signals, symbol)
        
        # 尝试执行交易
        trades = self._attempt_trading(symbol, data, signals)
        
        return {
            'symbol': symbol,
            'trades': trades,
            'signal_analysis': signal_analysis,
            'diagnostic_summary': self.signal_detector.get_diagnostic_summary()
        }
    
    def _analyze_signals(self, signals, symbol):
        """分析信号质量"""
        if signals.empty:
            logger.error(f"❌ {symbol}: 无信号生成!")
            return {}
        
        signal_types = signals['signal_type'].value_counts()
        avg_strength = signals['signal_strength'].mean()
        avg_confidence = signals['confidence'].mean()
        
        logger.info(f"📊 {symbol} 信号分析:")
        logger.info(f"  信号分布: {dict(signal_types)}")
        logger.info(f"  平均信号强度: {avg_strength:.3f}")
        logger.info(f"  平均置信度: {avg_confidence:.3f}")
        
        # 检查是否有强信号
        strong_signals = signals[signals['signal_type'].isin(['STRONG_BUY', 'STRONG_SELL'])]
        if strong_signals.empty:
            logger.warning(f"⚠️  {symbol}: 没有强信号!")
        else:
            logger.info(f"✅ {symbol}: 发现 {len(strong_signals)} 个强信号")
        
        return {
            'total_signals': len(signals),
            'strong_signals': len(strong_signals),
            'avg_strength': avg_strength,
            'avg_confidence': avg_confidence
        }
    
    def _attempt_trading(self, symbol, data, signals):
        """尝试执行交易"""
        trades = []
        
        if signals.empty:
            return trades
        
        # 查找强信号位置
        strong_indices = signals[signals['signal_type'].isin(['STRONG_BUY', 'STRONG_SELL'])].index
        
        for idx in strong_indices[:5]:  # 只尝试前5个强信号
            if idx < len(data):
                row = data.iloc[idx]
                signal = signals.iloc[idx]
                
                logger.info(f"💰 尝试交易: {signal['signal_type']} @ ${row['close']:.2f}")
                
                # 简单执行交易
                position_size = self.current_capital * 0.1
                
                if signal['signal_type'] == 'STRONG_BUY':
                    trade = {
                        'symbol': symbol, 'timestamp': row['timestamp'],
                        'action': 'BUY', 'price': row['close'],
                        'size': position_size, 'type': 'long'
                    }
                    trades.append(trade)
                    logger.info(f"✅ 执行买入交易")
                    
                elif signal['signal_type'] == 'STRONG_SELL':
                    trade = {
                        'symbol': symbol, 'timestamp': row['timestamp'],
                        'action': 'SELL', 'price': row['close'],
                        'size': position_size, 'type': 'short'
                    }
                    trades.append(trade)
                    logger.info(f"✅ 执行卖出交易")
        
        return trades
    
    def _generate_diagnostic_report(self, all_results):
        """生成诊断报告"""
        logger.info(f"\n{'='*80}")
        logger.info("🔍 深度诊断报告")
        logger.info(f"{'='*80}")
        
        for result in all_results:
            symbol = result['symbol']
            trades = result['trades']
            diagnostic_summary = result['diagnostic_summary']
            
            logger.info(f"\n📋 {symbol} 诊断结果:")
            logger.info(f"  交易次数: {len(trades)}")
            logger.info(f"  诊断摘要: {diagnostic_summary}")
            
            signal_analysis = result.get('signal_analysis', {})
            if signal_analysis:
                logger.info(f"  信号统计: {signal_analysis.get('total_signals', 0)}总信号, {signal_analysis.get('strong_signals', 0)}强信号")
                logger.info(f"  信号质量: 强度={signal_analysis.get('avg_strength', 0):.3f}, 置信度={signal_analysis.get('avg_confidence', 0):.3f}")
        
        # 总体建议
        total_trades = sum(len(r['trades']) for r in all_results)
        
        logger.info(f"\n💡 深度诊断建议:")
        if total_trades == 0:
            logger.info("🔴 严重问题: 系统完全没有产生交易")
            logger.info("   可能原因:")
            logger.info("   1. 信号生成逻辑错误")
            logger.info("   2. 技术指标计算问题") 
            logger.info("   3. 数据格式不匹配")
            logger.info("   4. 阈值设置过高")
        else:
            logger.info("🟢 系统基本正常，可以进一步优化")
        
        logger.info(f"\n🎯 深度诊断完成")

def main():
    parser = argparse.ArgumentParser(description='深度诊断版高频交易回测系统')
    parser.add_argument('--symbols', type=str, default='BTC/USDT')
    parser.add_argument('--days', type=int, default=10)
    parser.add_argument('--capital', type=float, default=10000)
    
    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    backtest = DiagnosticBacktest(initial_capital=args.capital)
    backtest.run_diagnostic_backtest(symbols=symbols, days=args.days)

if __name__ == "__main__":
    main()