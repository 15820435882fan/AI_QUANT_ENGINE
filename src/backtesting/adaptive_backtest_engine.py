# src/backtesting/adaptive_backtest_engine.py
#!/usr/bin/env python3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import asyncio

class AdaptiveBacktestEngine:
    """自适应回测引擎 - 测试多策略切换系统"""
    
    def __init__(self, config=None):
        from .backtest_engine import BacktestConfig
        self.config = config or BacktestConfig()
        
        # 动态导入以避免循环依赖
        from ..strategies.multi_strategy_manager import MultiStrategyManager
        from ..strategies.market_regime_detector import MarketRegimeDetector
        
        self.regime_detector = MarketRegimeDetector()
        self.strategy_manager = MultiStrategyManager()
        self.logger = logging.getLogger(__name__)
        
    async def run_adaptive_backtest(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """运行自适应策略回测"""
        print("🚀 开始自适应策略回测...")
        
        # 初始化状态
        balance = self.config.initial_capital
        positions: Dict[str, float] = {}
        trades: List[Dict] = []
        regime_history: List[Dict] = []
        strategy_performance: Dict[str, Dict] = {}
        
        # 滑动窗口分析 - 使用较小窗口加快测试
        window_size = 720  # 12小时的数据（720分钟）
        
        for i in range(window_size, len(historical_data), 120):  # 每2小时检测一次
            if i >= len(historical_data):
                break
                
            window_data = historical_data.iloc[i-window_size:i]
            current_data = historical_data.iloc[i]
            current_price = current_data['close']
            current_time = historical_data.index[i]
            
            try:
                # 1. 检测市场状态
                regime = await self.regime_detector.detect_regime(window_data)
                regime_history.append({
                    'timestamp': current_time,
                    'regime': regime,
                    'price': current_price
                })
                
                # 2. 更新策略选择
                await self.strategy_manager.update_market_regime(window_data)
                
                # 3. 模拟策略信号生成（简化版）
                active_strategies = self.strategy_manager.get_active_strategies()
                for strategy in active_strategies:
                    signal = await self._generate_signal(strategy, current_price, regime)
                    
                    if signal and self._should_execute_trade(signal, balance, positions):
                        # 执行交易
                        trade_result = await self._execute_trade(signal, current_price, balance, positions)
                        if trade_result:
                            trades.append(trade_result)
                            balance = trade_result['new_balance']
                            
                            # 记录策略表现
                            self._record_strategy_performance(
                                strategy_performance, 
                                strategy['name'], 
                                trade_result
                            )
                
            except Exception as e:
                self.logger.error(f"❌ 回测过程出错: {e}")
                continue
        
        # 计算最终结果
        final_price = historical_data['close'].iloc[-1]
        position_value = sum(positions.values()) * final_price
        final_equity = balance + position_value
        total_return = (final_equity - self.config.initial_capital) / self.config.initial_capital
        
        return {
            'total_return': total_return,
            'final_balance': final_equity,
            'total_trades': len(trades),
            'regime_changes': len(set([r['regime'] for r in regime_history])),
            'regime_history': regime_history,
            'strategy_performance': strategy_performance,
            'trades': trades
        }
    
    async def _generate_signal(self, strategy: Dict, current_price: float, regime: str) -> Optional[Dict]:
        """生成交易信号（简化版）"""
        import random
        
        # 基于策略类型和市场状态生成信号
        signal_probability = {
            'sma': 0.1 if regime in ['trending', 'strong_trend'] else 0.05,
            'rsi': 0.15 if regime in ['ranging', 'low_volatility'] else 0.05,
            'macd': 0.12 if regime in ['trending'] else 0.03,
            'bollinger': 0.1 if regime in ['ranging'] else 0.02
        }
        
        prob = signal_probability.get(strategy['type'], 0.05)
        if random.random() < prob:
            return {
                'strategy': strategy['name'],
                'action': 'buy' if random.random() > 0.5 else 'sell',
                'price': current_price,
                'strength': random.uniform(0.5, 0.9)
            }
        return None
    
    def _should_execute_trade(self, signal: Dict, balance: float, positions: Dict) -> bool:
        """判断是否执行交易"""
        # 简化版风险控制
        if signal['action'] == 'buy' and balance < 100:
            return False
        if signal['action'] == 'sell' and positions.get('BTC/USDT', 0) <= 0:
            return False
        return True
    
    async def _execute_trade(self, signal: Dict, current_price: float, balance: float, positions: Dict) -> Optional[Dict]:
        """执行交易"""
        symbol = 'BTC/USDT'
        quantity = 0.001  # 固定交易量
        
        if signal['action'] == 'buy':
            cost = quantity * current_price * (1 + self.config.commission)
            if cost <= balance:
                new_balance = balance - cost
                new_position = positions.get(symbol, 0) + quantity
                positions[symbol] = new_position
                
                return {
                    'timestamp': pd.Timestamp.now(),
                    'strategy': signal['strategy'],
                    'action': 'buy',
                    'price': current_price,
                    'quantity': quantity,
                    'new_balance': new_balance,
                    'profit': 0  # 买入时盈亏为0
                }
        else:  # sell
            current_position = positions.get(symbol, 0)
            if quantity <= current_position:
                revenue = quantity * current_price * (1 - self.config.commission)
                new_balance = balance + revenue
                new_position = current_position - quantity
                positions[symbol] = new_position
                
                # 简化盈亏计算（实际应该记录买入价格）
                profit = revenue - (quantity * current_price * 0.99)  # 假设1%利润
                
                return {
                    'timestamp': pd.Timestamp.now(),
                    'strategy': signal['strategy'],
                    'action': 'sell',
                    'price': current_price,
                    'quantity': quantity,
                    'new_balance': new_balance,
                    'profit': profit
                }
        return None
    
    def _record_strategy_performance(self, performance: Dict, strategy_name: str, trade: Dict):
        """记录策略表现"""
        if strategy_name not in performance:
            performance[strategy_name] = {
                'trades': 0,
                'total_profit': 0,
                'win_trades': 0
            }
        
        performance[strategy_name]['trades'] += 1
        profit = trade.get('profit', 0)
        performance[strategy_name]['total_profit'] += profit
        
        if profit > 0:
            performance[strategy_name]['win_trades'] += 1

# 测试函数
async def test_adaptive_engine():
    """测试自适应引擎"""
    from .backtest_engine import BacktestConfig, DataManager
    
    print("🧪 测试自适应回测引擎...")
    
    config = BacktestConfig(initial_capital=10000.0)
    engine = AdaptiveBacktestEngine(config)
    
    data_manager = DataManager()
    historical_data = await data_manager.load_historical_data(
        "BTC/USDT", "2024-01-01", "2024-01-05"
    )
    
    result = await engine.run_adaptive_backtest(historical_data)
    
    print(f"💰 总收益: {result['total_return']:.2%}")
    print(f"🔢 交易次数: {result['total_trades']}")
    print("✅ 自适应引擎测试完成")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_adaptive_engine())