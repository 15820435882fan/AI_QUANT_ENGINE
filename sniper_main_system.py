# sniper_main_system.py
import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入所有模块
from multi_exchange_monitor import SniperExchangeMonitor
from sniper_signal_detector import SniperSignalDetector
from sniper_position_manager import SniperPositionManager
from sniper_executor import SniperExecutor
from enhanced_compound_engine import EnhancedCompoundEngine

class CryptoSniperSystem:
    """加密货币刺客交易主控系统"""
    
    def __init__(self, capital: float = 10000.0, test_mode: bool = True):
        self.capital = capital
        self.test_mode = test_mode  # 测试模式，不真实交易
        
        # 初始化所有组件
        self.monitor = SniperExchangeMonitor()
        self.signal_detector = SniperSignalDetector()
        self.position_manager = SniperPositionManager(capital)
        
        # 只有在非测试模式才初始化真实交易执行器
        if not test_mode:
            self.executor = SniperExecutor()
        
        # 复利引擎用于长期策略
        self.compound_engine = EnhancedCompoundEngine(capital * 0.3)  # 30%资金用于复利
        
        # 交易记录和性能追踪
        self.trade_history = []
        self.performance_data = []
        self.active_positions = {}
        
        self.setup_logging()
        self.setup_strategies()
        
        logging.info(f"🎯 加密货币刺客系统初始化完成 - 资金: ${capital:,.2f}")
    
    def setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('sniper_system.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('CryptoSniper')
    
    def setup_strategies(self):
        """设置交易策略"""
        from src.strategies.trend_following_enhanced import TrendFollowingEnhanced
        from src.strategies.mean_reversion_compound import MeanReversionCompound
        
        # 为复利引擎添加策略
        trend_strategy = TrendFollowingEnhanced({
            'name': '刺客趋势',
            'weight': 0.6,
            'parameters': {'fast_window': 5, 'slow_window': 15, 'momentum_window': 8}
        })
        
        mean_reversion_strategy = MeanReversionCompound({
            'name': '刺客均值回归', 
            'weight': 0.4,
            'parameters': {'bb_period': 10, 'bb_std': 1.5}
        })
        
        self.compound_engine.add_strategy(trend_strategy)
        self.compound_engine.add_strategy(mean_reversion_strategy)
        
        self.logger.info("✅ 交易策略设置完成")
    
    async def monitor_market_opportunities(self) -> List[Dict]:
        """监控市场机会 - 核心循环"""
        all_opportunities = []
        
        self.logger.info("🔍 开始市场机会扫描...")
        
        for symbol in self.monitor.symbols:
            try:
                # 监控异常波动
                alerts = await self.monitor.monitor_volume_spike(symbol)
                
                for alert in alerts:
                    # 获取详细K线数据进行分析
                    exchange = self.monitor.exchanges[alert['exchange']]
                    ohlcv = exchange.fetch_ohlcv(alert['symbol'], '5m', limit=100)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # 信号确认
                    signal = self.signal_detector.confirm_sniper_signal(alert, df)
                    
                    if signal['confirmed']:
                        opportunity = {
                            'symbol': alert['symbol'],
                            'exchange': alert['exchange'],
                            'signal': signal,
                            'data': df,
                            'timestamp': datetime.now()
                        }
                        all_opportunities.append(opportunity)
                        
                        self.logger.info(f"🎯 发现交易机会: {signal['direction']} {alert['symbol']} "
                                      f"置信度: {signal['confidence']:.2f} "
                                      f"量比: {alert['volume_ratio']:.1f}x")
                
            except Exception as e:
                self.logger.error(f"监控{symbol}时出错: {e}")
        
        return all_opportunities
    
    def evaluate_opportunity_quality(self, opportunity: Dict) -> Dict[str, Any]:
        """评估机会质量"""
        signal = opportunity['signal']
        df = opportunity['data']
        
        # 技术指标深度分析
        quality_score = signal['confidence']
        
        # 成交量确认
        volume_trend = df['volume'].tail(5).mean() / df['volume'].tail(20).mean()
        if volume_trend > 1.5:
            quality_score *= 1.2
        
        # 价格动量确认
        price_momentum = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
        if abs(price_momentum) > 0.03:
            quality_score *= 1.1
        
        # 市场环境考量
        market_regime = self._analyze_market_regime(df)
        if market_regime == signal['direction'].lower():
            quality_score *= 1.15
        
        return {
            'quality_score': min(quality_score, 1.0),
            'volume_trend': volume_trend,
            'price_momentum': price_momentum,
            'market_regime': market_regime
        }
    
    def _analyze_market_regime(self, df: pd.DataFrame) -> str:
        """分析市场状态"""
        price_change_5m = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6]
        price_change_1h = (df['close'].iloc[-1] - df['close'].iloc[-12]) / df['close'].iloc[-12]
        
        if price_change_1h > 0.02 and price_change_5m > 0.005:
            return 'bullish'
        elif price_change_1h < -0.02 and price_change_5m < -0.005:
            return 'bearish'
        else:
            return 'neutral'
    
    async def execute_sniper_trade(self, opportunity: Dict, evaluation: Dict):
        """执行刺客交易"""
        try:
            signal = opportunity['signal']
            symbol = opportunity['symbol']
            
            # 计算仓位
            position = self.position_manager.calculate_position_size(signal)
            
            # 记录交易决策
            trade_decision = {
                'symbol': symbol,
                'exchange': opportunity['exchange'],
                'direction': signal['direction'],
                'entry_price': signal['entry_price'],
                'position_size': position['position_size'],
                'leverage': position['leverage'],
                'quantity': position['quantity'],
                'stop_loss': position['stop_loss'],
                'take_profit': position['take_profit'],
                'confidence': signal['confidence'],
                'quality_score': evaluation['quality_score'],
                'timestamp': datetime.now(),
                'status': 'PENDING'
            }
            
            self.logger.info(f"💸 交易决策: {signal['direction']} {symbol} "
                          f"仓位: ${position['position_size']} "
                          f"杠杆: {position['leverage']}x "
                          f"止损: {position['stop_loss']:.2f}")
            
            # 执行交易（测试模式只记录不真实交易）
            if not self.test_mode:
                trade_result = await self.executor.execute_sniper_trade(signal, position)
                trade_decision.update(trade_result)
                trade_decision['status'] = 'EXECUTED'
            else:
                trade_decision['status'] = 'TEST_MODE'
                self.logger.info("🧪 测试模式 - 未执行真实交易")
            
            # 记录交易
            self.trade_history.append(trade_decision)
            self.active_positions[symbol] = trade_decision
            
            return trade_decision
            
        except Exception as e:
            self.logger.error(f"交易执行失败: {e}")
            return None
    
    async def run_compound_engine(self):
        """运行复利引擎"""
        try:
            self.logger.info("🔄 运行复利引擎...")
            
            # 为每个币种生成复利信号
            for symbol in self.monitor.symbols[:2]:  # 只处理前两个币种
                exchange = self.monitor.exchanges['binance']
                ohlcv = exchange.fetch_ohlcv(symbol, '15m', limit=100)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # 生成复利信号
                compound_signals = self.compound_engine.generate_compound_signals(df)
                
                if 'error' not in compound_signals:
                    self.logger.info(f"📊 复利信号 {symbol}: {compound_signals['decision']['action']} "
                                  f"置信度: {compound_signals['combined_confidence']:.2f}")
        
        except Exception as e:
            self.logger.error(f"复利引擎运行错误: {e}")
    
    async def run_daily_cycle(self):
        """运行每日交易周期"""
        self.logger.info("🚀 开始每日交易周期...")
        
        cycle_count = 0
        max_cycles = 288  # 24小时 * 12次/小时 (5分钟间隔)
        
        while cycle_count < max_cycles:
            try:
                cycle_count += 1
                self.logger.info(f"🔄 交易周期 #{cycle_count}")
                
                # 1. 监控市场机会
                opportunities = await self.monitor_market_opportunities()
                
                # 2. 评估和执行最佳机会
                for opportunity in opportunities:
                    evaluation = self.evaluate_opportunity_quality(opportunity)
                    
                    # 只执行高质量机会
                    if evaluation['quality_score'] > 0.75:
                        await self.execute_sniper_trade(opportunity, evaluation)
                
                # 3. 每6个周期运行一次复利引擎
                if cycle_count % 6 == 0:
                    await self.run_compound_engine()
                
                # 4. 显示系统状态
                if cycle_count % 12 == 0:
                    self.display_system_status()
                
                # 等待5分钟进行下一轮监控
                self.logger.info("⏳ 等待5分钟进行下一轮监控...")
                await asyncio.sleep(300)  # 5分钟
                
            except Exception as e:
                self.logger.error(f"交易周期错误: {e}")
                await asyncio.sleep(60)  # 出错等待1分钟
        
        self.logger.info("✅ 每日交易周期完成")
    
    def display_system_status(self):
        """显示系统状态"""
        active_trades = len(self.active_positions)
        total_trades = len(self.trade_history)
        profitable_trades = len([t for t in self.trade_history if t.get('profit_loss', 0) > 0])
        
        print(f"\n{'='*60}")
        print(f"🎯 刺客交易系统状态")
        print(f"{'='*60}")
        print(f"📊 活跃交易: {active_trades}")
        print(f"📈 总交易数: {total_trades}")
        print(f"✅ 盈利交易: {profitable_trades}")
        print(f"💰 剩余资金: ${self.capital:,.2f}")
        
        if self.active_positions:
            print(f"\n📦 当前持仓:")
            for symbol, position in self.active_positions.items():
                print(f"   {symbol}: {position['direction']} ${position['position_size']} "
                      f"杠杆{position['leverage']}x")
        
        print(f"{'='*60}\n")
    
    async def run_system(self, days: int = 1):
        """运行主系统"""
        self.logger.info(f"🚀 启动加密货币刺客系统 - 运行{days}天")
        
        for day in range(1, days + 1):
            self.logger.info(f"📅 第{day}天开始")
            
            await self.run_daily_cycle()
            
            # 每日总结
            self.daily_summary(day)
            
            if day < days:
                self.logger.info("🌙 每日结束，等待第二天...")
                await asyncio.sleep(2)  # 模拟过夜
        
        self.generate_final_report()
    
    def daily_summary(self, day: int):
        """每日总结"""
        day_trades = [t for t in self.trade_history 
                     if t['timestamp'].date() == datetime.now().date()]
        
        if day_trades:
            day_profit = sum(t.get('profit_loss', 0) for t in day_trades)
            self.capital += day_profit
            
            self.logger.info(f"📊 第{day}天总结: "
                          f"交易{len(day_trades)}次, "
                          f"当日盈亏: ${day_profit:+.2f}, "
                          f"总资金: ${self.capital:,.2f}")
    
    def generate_final_report(self):
        """生成最终报告"""
        print(f"\n{'='*80}")
        print(f"🎉 加密货币刺客系统 - 最终报告")
        print(f"{'='*80}")
        
        total_trades = len(self.trade_history)
        profitable_trades = len([t for t in self.trade_history if t.get('profit_loss', 0) > 0])
        total_profit = sum(t.get('profit_loss', 0) for t in self.trade_history)
        
        print(f"📈 总交易次数: {total_trades}")
        print(f"✅ 盈利交易: {profitable_trades}")
        print(f"❌ 亏损交易: {total_trades - profitable_trades}")
        print(f"🎯 胜率: {profitable_trades/total_trades*100:.1f}%" if total_trades > 0 else "N/A")
        print(f"💰 总盈亏: ${total_profit:+.2f}")
        print(f"📊 最终资金: ${self.capital:,.2f}")
        print(f"📉 资金增长率: {(self.capital - 10000)/10000*100:+.2f}%")
        
        # 显示最近交易
        if self.trade_history:
            print(f"\n📋 最近5笔交易:")
            for trade in self.trade_history[-5:]:
                status = "盈利" if trade.get('profit_loss', 0) > 0 else "亏损"
                print(f"   {trade['symbol']} {trade['direction']} | "
                      f"${trade['position_size']} | {status}")

# 测试运行函数
async def test_system():
    """测试系统运行"""
    print("🧪 测试加密货币刺客系统...")
    
    # 创建系统实例（测试模式）
    sniper_system = CryptoSniperSystem(capital=10000.0, test_mode=True)
    
    # 运行1天测试
    await sniper_system.run_system(days=1)
    
    return sniper_system

# 主运行函数
async def main():
    """主运行函数"""
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        await test_system()
    else:
        # 真实运行（需要配置API密钥）
        sniper_system = CryptoSniperSystem(capital=10000.0, test_mode=False)
        await sniper_system.run_system(days=7)  # 运行7天

if __name__ == "__main__":
    # 默认运行测试模式
    asyncio.run(test_system())