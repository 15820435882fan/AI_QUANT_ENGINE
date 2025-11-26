# src/trading/live_trader.py (修复版本)
#!/usr/bin/env python3
import asyncio
import logging
import pandas as pd
from typing import Dict, Optional

class LiveTrader:
    """实盘交易引擎"""
    
    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        
        # 初始化组件
        from src.strategies.multi_strategy_manager import MultiStrategyManager
        from src.strategies.market_regime_detector import MarketRegimeDetector
        from src.risk.risk_manager import RiskManager, RiskConfig
        
        self.strategy_manager = MultiStrategyManager()
        self.regime_detector = MarketRegimeDetector()
        self.risk_manager = RiskManager(RiskConfig())
        
        # 模拟数据缓存
        self.mock_data = self._generate_mock_data()
        
    def _generate_mock_data(self) -> pd.DataFrame:
        """生成模拟数据用于测试"""
        dates = pd.date_range(start="2024-01-01", periods=100, freq='1min')
        data = []
        price = 50000.0
        
        for date in dates:
            change = np.random.normal(0, 0.001)
            price = price * (1 + change)
            
            data.append({
                'timestamp': date,
                'open': price,
                'high': price * 1.001,
                'low': price * 0.999,
                'close': price,
                'volume': np.random.uniform(1000, 5000)
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
        
    async def start_trading(self):
        """开始实盘交易"""
        self.is_running = True
        self.logger.info("🚀 启动实盘交易系统...")
        
        if self.paper_trading:
            self.logger.info("📝 当前为模拟交易模式")
        else:
            self.logger.info("💰 当前为实盘交易模式 - 请谨慎!")
        
        # 主交易循环
        cycle_count = 0
        while self.is_running and cycle_count < 10:  # 限制循环次数用于测试
            try:
                await self._trading_cycle(cycle_count)
                cycle_count += 1
                await asyncio.sleep(5)  # 5秒一次，加快测试
                
            except Exception as e:
                self.logger.error(f"❌ 交易循环出错: {e}")
                await asyncio.sleep(2)
    
    async def _trading_cycle(self, cycle_count: int):
        """交易周期"""
        # 1. 获取市场数据 - 使用模拟数据
        market_data = await self._get_market_data(cycle_count)
        
        # 2. 检测市场状态
        regime = await self.regime_detector.detect_regime(self.mock_data)
        self.logger.info(f"📊 市场状态: {regime}")
        
        # 3. 更新策略选择
        await self.strategy_manager.update_market_regime(self.mock_data)
        
        # 4. 生成交易信号
        active_strategies = self.strategy_manager.get_active_strategies()
        self.logger.info(f"🎯 激活策略: {len(active_strategies)}个")
        
        # 5. 模拟交易信号
        for strategy in active_strategies:
            signal = await self._generate_mock_signal(strategy, market_data)
            if signal:
                await self._execute_trade(signal, strategy['name'])
    
    async def _get_market_data(self, cycle_count: int) -> Dict:
        """获取市场数据"""
        # 使用模拟数据
        if cycle_count < len(self.mock_data):
            row = self.mock_data.iloc[cycle_count]
            return {
                'symbol': 'BTC/USDT',
                'price': float(row['close']),
                'timestamp': row.name.timestamp()
            }
        else:
            return {
                'symbol': 'BTC/USDT',
                'price': 50000.0,
                'timestamp': asyncio.get_event_loop().time()
            }
    
    async def _generate_mock_signal(self, strategy: Dict, market_data: Dict) -> Optional[Dict]:
        """生成模拟交易信号"""
        import random
        
        # 10%概率生成信号
        if random.random() < 0.1:
            return {
                'action': 'buy' if random.random() > 0.5 else 'sell',
                'price': market_data['price'],
                'strength': random.uniform(0.5, 0.9)
            }
        return None
    
    async def _execute_trade(self, signal: Dict, strategy_name: str):
        """执行交易"""
        try:
            # 风险检查
            risk_result = await self.risk_manager.validate_trade(
                signal, 
                10000.0,  # 当前权益
                {},       # 当前持仓
                0         # 今日交易数
            )
            
            if not risk_result['approved']:
                self.logger.warning(f"⛔ 交易被风控拒绝: {risk_result['reason']}")
                return
            
            # 执行交易
            if self.paper_trading:
                self.logger.info(f"📝 模拟交易: {strategy_name} "
                               f"{signal['action']} @ {signal['price']:.2f}")
            else:
                self.logger.info(f"💰 实盘交易: {strategy_name} "
                               f"{signal['action']}")
                
        except Exception as e:
            self.logger.error(f"❌ 交易执行失败: {e}")
    
    async def stop_trading(self):
        """停止交易"""
        self.is_running = False
        self.logger.info("🛑 停止交易系统")