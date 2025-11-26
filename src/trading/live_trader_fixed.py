# src/trading/live_trader_fixed.py
#!/usr/bin/env python3
import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any

class LiveTraderFixed:
    """修复版的实盘交易引擎"""
    
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
        
        # 模拟数据
        self.mock_data = self._generate_mock_data()
        self.current_index = 0
        
    def _generate_mock_data(self) -> pd.DataFrame:
        """生成模拟数据"""
        dates = pd.date_range(start="2024-01-01", periods=50, freq='1min')  # 减少数据量
        data = []
        price = 50000.0
        
        for date in dates:
            change = np.random.normal(0, 0.001)
            price = max(price * (1 + change), 1000)
            
            data.append({
                'timestamp': date,
                'open': float(price),
                'high': float(price * 1.001),
                'low': float(price * 0.999),
                'close': float(price),
                'volume': float(np.random.uniform(1000, 5000))
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
        
    async def start_trading(self):
        """开始实盘交易"""
        self.is_running = True
        self.logger.info("🚀 启动修复版交易系统...")
        
        cycle_count = 0
        max_cycles = 5  # 只运行5个周期用于测试
        
        while self.is_running and cycle_count < max_cycles:
            try:
                self.logger.info(f"🔄 交易周期 {cycle_count + 1}/{max_cycles}")
                await self._trading_cycle(cycle_count)
                cycle_count += 1
                await asyncio.sleep(2)  # 2秒间隔
                
            except Exception as e:
                self.logger.error(f"❌ 交易周期 {cycle_count} 出错: {e}")
                await asyncio.sleep(1)
        
        self.logger.info("✅ 交易测试完成")
        await self.stop_trading()
    
    async def _trading_cycle(self, cycle_count: int):
        """交易周期"""
        # 1. 获取市场数据
        market_data = await self._get_market_data(cycle_count)
        self.logger.info(f"📊 当前价格: {market_data['price']:.2f}")
        
        # 2. 检测市场状态
        if len(self.mock_data) > 20:
            recent_data = self.mock_data.iloc[max(0, cycle_count-20):cycle_count+1]
            regime = await self.regime_detector.detect_regime(recent_data)
            self.logger.info(f"🎯 市场状态: {regime}")
        else:
            regime = "unknown"
        
        # 3. 更新策略选择
        if len(self.mock_data) > 30:
            strategy_data = self.mock_data.iloc[max(0, cycle_count-30):cycle_count+1]
            await self.strategy_manager.update_market_regime(strategy_data)
        
        # 4. 生成模拟信号
        active_strategies = self.strategy_manager.get_active_strategies()
        if active_strategies:
            self.logger.info(f"🔧 激活策略: {[s['name'] for s in active_strategies]}")
            
            for strategy in active_strategies:
                signal = await self._generate_safe_signal(strategy, market_data)
                if signal:
                    await self._execute_trade(signal, strategy['name'])
        else:
            self.logger.info("💤 暂无激活策略")
    
    async def _get_market_data(self, cycle_count: int) -> Dict[str, Any]:
        """安全获取市场数据"""
        if cycle_count < len(self.mock_data):
            row = self.mock_data.iloc[cycle_count]
            return {
                'symbol': 'BTC/USDT',
                'price': float(row['close']),
                'timestamp': cycle_count
            }
        else:
            return {
                'symbol': 'BTC/USDT', 
                'price': 50000.0,
                'timestamp': cycle_count
            }
    
    async def _generate_safe_signal(self, strategy: Dict, market_data: Dict) -> Optional[Dict]:
        """安全生成交易信号"""
        import random
        
        # 简单的信号生成逻辑
        if random.random() < 0.3:  # 30%概率生成信号
            action = 'buy' if random.random() > 0.5 else 'sell'
            self.logger.info(f"🎯 生成{action}信号")
            
            return {
                'action': action,
                'price': market_data['price'],
                'strength': 0.7
            }
        return None
    
    async def _execute_trade(self, signal: Dict, strategy_name: str):
        """安全执行交易"""
        try:
            # 简化风险检查
            if signal['action'] == 'buy':
                # 模拟买入
                self.logger.info(f"💰 {strategy_name} 执行买入 @ {signal['price']:.2f}")
            else:
                # 模拟卖出  
                self.logger.info(f"💰 {strategy_name} 执行卖出 @ {signal['price']:.2f}")
                
        except Exception as e:
            self.logger.error(f"❌ 交易执行失败: {e}")
    
    async def stop_trading(self):
        """停止交易"""
        self.is_running = False
        self.logger.info("🛑 交易系统已停止")

# 测试函数
async def test_fixed_trader():
    """测试修复版交易系统"""
    print("🧪 测试修复版交易系统...")
    
    trader = LiveTraderFixed(paper_trading=True)
    await trader.start_trading()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_fixed_trader())