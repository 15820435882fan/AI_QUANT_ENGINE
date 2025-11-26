# high_performance_strategies.py
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import asyncio

from src.strategies.strategy_orchestrator import BaseStrategy, TradingSignal, SignalType

class RSIMomentumStrategy(BaseStrategy):
    """RSI动量策略 - 简单有效"""
    
    def __init__(self, name: str, symbols: List[str], rsi_period: int = 14, 
                 oversold: int = 30, overbought: int = 70):
        super().__init__()
        self.name = name
        self.symbols = symbols
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.price_history = []
        
    async def analyze(self, market_data) -> Optional[TradingSignal]:
        try:
            current_price = getattr(market_data, 'close', 0)
            self.price_history.append(current_price)
            
            # 需要足够的数据计算RSI
            if len(self.price_history) < self.rsi_period + 1:
                return None
            
            # 计算RSI
            prices = pd.Series(self.price_history[-self.rsi_period-1:])
            rsi = self.calculate_rsi(prices)
            
            if rsi < self.oversold:
                return TradingSignal(
                    symbol=getattr(market_data, 'symbol', 'BTC/USDT'),
                    signal_type=SignalType.BUY,
                    price=current_price,
                    strength=0.8,
                    timestamp=getattr(market_data, 'timestamp', datetime.now().timestamp()),
                    reason=f"RSI超卖 ({rsi:.1f} < {self.oversold})"
                )
            elif rsi > self.overbought:
                return TradingSignal(
                    symbol=getattr(market_data, 'symbol', 'BTC/USDT'),
                    signal_type=SignalType.SELL,
                    price=current_price,
                    strength=0.8,
                    timestamp=getattr(market_data, 'timestamp', datetime.now().timestamp()),
                    reason=f"RSI超买 ({rsi:.1f} > {self.overbought})"
                )
            
            return None
            
        except Exception as e:
            print(f"RSI策略错误: {e}")
            return None
    
    def calculate_rsi(self, prices: pd.Series) -> float:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

class EMACrossoverStrategy(BaseStrategy):
    """EMA双均线策略 - 经典趋势跟踪"""
    
    def __init__(self, name: str, symbols: List[str], fast_period: int = 9, slow_period: int = 21):
        super().__init__()
        self.name = name
        self.symbols = symbols
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.price_history = []
        self.current_position = None
        
    async def analyze(self, market_data) -> Optional[TradingSignal]:
        try:
            current_price = getattr(market_data, 'close', 0)
            self.price_history.append(current_price)
            
            if len(self.price_history) < self.slow_period:
                return None
            
            # 计算EMA
            prices = pd.Series(self.price_history)
            fast_ema = prices.ewm(span=self.fast_period).mean().iloc[-1]
            slow_ema = prices.ewm(span=self.slow_period).mean().iloc[-1]
            
            signal = None
            
            # 金叉买入
            if fast_ema > slow_ema and self.current_position != "long":
                signal = TradingSignal(
                    symbol=getattr(market_data, 'symbol', 'BTC/USDT'),
                    signal_type=SignalType.BUY,
                    price=current_price,
                    strength=0.7,
                    timestamp=getattr(market_data, 'timestamp', datetime.now().timestamp()),
                    reason=f"EMA金叉 ({fast_ema:.2f} > {slow_ema:.2f})"
                )
                self.current_position = "long"
            
            # 死叉卖出
            elif fast_ema < slow_ema and self.current_position != "short":
                signal = TradingSignal(
                    symbol=getattr(market_data, 'symbol', 'BTC/USDT'),
                    signal_type=SignalType.SELL,
                    price=current_price,
                    strength=0.7,
                    timestamp=getattr(market_data, 'timestamp', datetime.now().timestamp()),
                    reason=f"EMA死叉 ({fast_ema:.2f} < {slow_ema:.2f})"
                )
                self.current_position = "short"
            
            return signal
            
        except Exception as e:
            print(f"EMA策略错误: {e}")
            return None

class BreakoutStrategy(BaseStrategy):
    """突破策略 - 捕捉关键价位突破"""
    
    def __init__(self, name: str, symbols: List[str], period: int = 20):
        super().__init__()
        self.name = name
        self.symbols = symbols
        self.period = period
        self.price_history = []
        self.current_position = None
        
    async def analyze(self, market_data) -> Optional[TradingSignal]:
        try:
            current_price = getattr(market_data, 'close', 0)
            high_price = getattr(market_data, 'high', current_price * 1.01)
            low_price = getattr(market_data, 'low', current_price * 0.99)
            
            self.price_history.append({
                'high': high_price,
                'low': low_price,
                'close': current_price
            })
            
            if len(self.price_history) < self.period:
                return None
            
            # 计算阻力位和支撑位
            recent_highs = [p['high'] for p in self.price_history[-self.period:]]
            recent_lows = [p['low'] for p in self.price_history[-self.period:]]
            
            resistance = max(recent_highs)
            support = min(recent_lows)
            
            signal = None
            
            # 突破阻力位买入
            if current_price > resistance and self.current_position != "long":
                signal = TradingSignal(
                    symbol=getattr(market_data, 'symbol', 'BTC/USDT'),
                    signal_type=SignalType.BUY,
                    price=current_price,
                    strength=0.9,
                    timestamp=getattr(market_data, 'timestamp', datetime.now().timestamp()),
                    reason=f"突破阻力位 {resistance:.2f}"
                )
                self.current_position = "long"
            
            # 跌破支撑位卖出
            elif current_price < support and self.current_position != "short":
                signal = TradingSignal(
                    symbol=getattr(market_data, 'symbol', 'BTC/USDT'),
                    signal_type=SignalType.SELL,
                    price=current_price,
                    strength=0.9,
                    timestamp=getattr(market_data, 'timestamp', datetime.now().timestamp()),
                    reason=f"跌破支撑位 {support:.2f}"
                )
                self.current_position = "short"
            
            return signal
            
        except Exception as e:
            print(f"突破策略错误: {e}")
            return None

class SimpleMultiStrategyManager:
    """简化版多策略管理器 - 专注于有效策略"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        
        # 使用简单有效的策略组合
        self.strategies = {
            "rsi": RSIMomentumStrategy("RSI动量", symbols, rsi_period=14, oversold=30, overbought=70),
            "ema": EMACrossoverStrategy("EMA交叉", symbols, fast_period=9, slow_period=21),
            "breakout": BreakoutStrategy("突破策略", symbols, period=20)
        }
        
        self.signal_history = []
        
    async def analyze(self, market_data) -> Optional[TradingSignal]:
        """多策略分析"""
        all_signals = []
        symbol = getattr(market_data, 'symbol', 'BTC/USDT')
        
        # 并行运行所有策略
        for strategy_name, strategy in self.strategies.items():
            try:
                signal = await strategy.analyze(market_data)
                if signal:
                    all_signals.append(signal)
                    print(f"✅ {strategy_name}: {signal.signal_type.value} - {signal.reason}")
            except Exception as e:
                print(f"❌ {strategy_name} 错误: {e}")
        
        # 简单多数投票
        if all_signals:
            buy_count = sum(1 for s in all_signals if s.signal_type == SignalType.BUY)
            sell_count = sum(1 for s in all_signals if s.signal_type == SignalType.SELL)
            
            if buy_count > sell_count:
                final_signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    price=all_signals[0].price,
                    strength=min(buy_count / len(all_signals), 1.0),
                    timestamp=all_signals[0].timestamp,
                    reason=f"多策略共识买入 ({buy_count}/{len(all_signals)})"
                )
            elif sell_count > buy_count:
                final_signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=all_signals[0].price,
                    strength=min(sell_count / len(all_signals), 1.0),
                    timestamp=all_signals[0].timestamp,
                    reason=f"多策略共识卖出 ({sell_count}/{len(all_signals)})"
                )
            else:
                return None
            
            self.signal_history.append(final_signal)
            return final_signal
        
        return None

# 测试函数
async def test_high_performance_strategies():
    """测试高性能策略"""
    print("🚀 测试高性能量化策略")
    print("=" * 50)
    
    # 创建测试数据 - 有明显趋势和波动的数据
    def create_test_data():
        np.random.seed(42)
        prices = [50000]
        
        # 创建明显的上升趋势 + 波动
        for i in range(200):
            # 基础趋势
            trend = 0.001  # 轻微上升趋势
            
            # 周期性波动
            cycle = 0.005 * np.sin(i * 0.1)
            
            # 随机噪声
            noise = np.random.normal(0, 0.008)
            
            change = trend + cycle + noise
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        return prices
    
    test_prices = create_test_data()
    manager = SimpleMultiStrategyManager(["BTC/USDT"])
    
    print(f"📊 测试数据: {len(test_prices)} 个价格点")
    print(f"📈 价格范围: {min(test_prices):.2f} - {max(test_prices):.2f}")
    
    signals = []
    
    for i, price in enumerate(test_prices):
        # 创建市场数据
        class MarketData:
            def __init__(self, price, timestamp):
                self.symbol = "BTC/USDT"
                self.close = price
                self.high = price * 1.005  # 模拟高价
                self.low = price * 0.995   # 模拟低价
                self.timestamp = timestamp
        
        timestamp = datetime.now().timestamp() + i * 3600  # 每小时一个数据点
        market_data = MarketData(price, timestamp)
        
        signal = await manager.analyze(market_data)
        
        if signal:
            signals.append(signal)
            print(f"🎯 最终信号 #{len(signals)}: {signal.signal_type.value} @ {signal.price:.2f}")
            print(f"   原因: {signal.reason}")
            print("---")
    
    print(f"\n📊 测试结果:")
    print(f"   总信号数: {len(signals)}")
    print(f"   信号频率: {len(signals)/len(test_prices)*100:.2f}%")
    
    if signals:
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        print(f"   买入信号: {len(buy_signals)}")
        print(f"   卖出信号: {len(sell_signals)}")
        
        # 简单收益计算
        if len(signals) >= 2:
            first_price = signals[0].price
            last_price = signals[-1].price
            profit_pct = (last_price - first_price) / first_price * 100
            print(f"   模拟收益: {profit_pct:+.2f}%")
    
    return len(signals) > 5

if __name__ == "__main__":
    success = asyncio.run(test_high_performance_strategies())
    if success:
        print("\n🎉 高性能策略测试成功！系统现在能生成足够的交易信号。")
    else:
        print("\n⚠️ 信号仍然较少，建议检查策略参数。")