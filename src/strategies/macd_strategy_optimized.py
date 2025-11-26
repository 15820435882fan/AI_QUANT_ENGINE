# macd_strategy_optimized.py
#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import asyncio
from collections import deque
from enum import Enum
from dataclasses import dataclass

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

try:
    from src.strategies.strategy_orchestrator import BaseStrategy, TradingSignal, SignalType
    from src.data.data_pipeline import MarketData, DataType
except ImportError:
    # 兼容性定义
    class SignalType(Enum):
        BUY = "buy"
        SELL = "sell"
        HOLD = "hold"
    
    @dataclass
    class TradingSignal:
        symbol: str
        signal_type: SignalType
        strength: float
        price: float
        timestamp: float
        reason: str = ""
        metadata: Dict = None
        
        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}
    
    class BaseStrategy:
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.name = config.get('name', 'Unnamed Strategy')
    
    class MarketData:
        def __init__(self, symbol: str, data: Any, timestamp: float, data_type: str = "OHLCV"):
            self.symbol = symbol
            self.data = data
            self.timestamp = timestamp
            self.data_type = data_type

# 导入数据兼容层
try:
    from data_compatibility import data_comp
except ImportError:
    # 内联兼容层
    class DataCompatibility:
        @staticmethod
        def get_close_price(market_data) -> float:
            try:
                if hasattr(market_data, 'close'):
                    return float(market_data.close)
                elif hasattr(market_data, 'data'):
                    data = market_data.data
                    if isinstance(data, (list, tuple)) and len(data) >= 5:
                        return float(data[4])
                    elif isinstance(data, dict) and 'close' in data:
                        return float(data['close'])
                elif hasattr(market_data, 'price'):
                    return float(market_data.price)
            except (ValueError, TypeError, IndexError) as e:
                print(f"收盘价提取错误: {e}")
            return None
        
        @staticmethod
        def get_high_price(market_data) -> float:
            try:
                if hasattr(market_data, 'high'):
                    return float(market_data.high)
                elif hasattr(market_data, 'data'):
                    data = market_data.data
                    if isinstance(data, (list, tuple)) and len(data) >= 4:
                        return float(data[2])
                    elif isinstance(data, dict) and 'high' in data:
                        return float(data['high'])
            except (ValueError, TypeError, IndexError) as e:
                print(f"最高价提取错误: {e}")
            return None
    
    data_comp = DataCompatibility()

class MACDStrategyOptimized(BaseStrategy):
    """MACD策略优化版 - 改进信号条件和风险管理"""
    
    def __init__(self, name: str, symbols: List[str], 
                 fast_period: int = 12, 
                 slow_period: int = 26, 
                 signal_period: int = 9,
                 min_trend_strength: float = 0.001,
                 hist_threshold: float = 0.0001,
                 min_trade_interval: int = 3):
        
        config = {
            'name': name,
            'symbols': symbols,
            'parameters': {
                'fast_period': fast_period,
                'slow_period': slow_period,
                'signal_period': signal_period,
                'min_trend_strength': min_trend_strength,
                'hist_threshold': hist_threshold,
                'min_trade_interval': min_trade_interval
            }
        }
        super().__init__(config)
        
        # 策略参数
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.min_trend_strength = min_trend_strength
        self.hist_threshold = hist_threshold
        self.min_trade_interval = min_trade_interval
        
        # 数据存储
        self.price_data = {symbol: deque(maxlen=slow_period + signal_period + 20) for symbol in symbols}
        self.macd_history = {symbol: deque(maxlen=10) for symbol in symbols}
        
        # 状态跟踪
        self.name = name
        self.signal_count = 0
        self.last_signal_time = {symbol: 0 for symbol in symbols}
        self.previous_signals = {symbol: [] for symbol in symbols}
    
    async def analyze(self, market_data) -> Optional[TradingSignal]:
        """优化后的MACD策略分析"""
        symbol = market_data.symbol
        
        # 提取收盘价
        close_price = data_comp.get_close_price(market_data)
        if close_price is None:
            return None
        
        # 更新价格数据
        self.price_data[symbol].append(close_price)
        
        # 检查最小数据长度
        min_data_length = self.slow_period + self.signal_period
        if len(self.price_data[symbol]) < min_data_length:
            print(f"📈 数据积累中: {len(self.price_data[symbol])}/{min_data_length}")
            return None
        
        # 计算MACD指标
        macd_line, signal_line, histogram = self._calculate_macd_optimized(symbol)
        if not macd_line or len(macd_line) == 0:
            return None
        
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        current_histogram = histogram[-1]
        
        # 更新MACD历史
        self.macd_history[symbol].append({
            'macd': current_macd,
            'signal': current_signal,
            'histogram': current_histogram,
            'price': close_price,
            'timestamp': market_data.timestamp
        })
        
        # 调试信息
        self._print_debug_info(symbol, close_price, current_macd, current_signal, current_histogram, histogram)
        
        # 生成交易信号
        signal = self._generate_signal(symbol, close_price, market_data.timestamp, 
                                     current_macd, current_signal, current_histogram, histogram)
        
        return signal
    
    def _calculate_macd_optimized(self, symbol: str) -> Tuple[List[float], List[float], List[float]]:
        """优化的MACD计算 - 使用固定窗口"""
        try:
            prices = list(self.price_data[symbol])
            
            if len(prices) < self.slow_period:
                return [], [], []
            
            # 使用pandas计算
            price_series = pd.Series(prices)
            
            # 计算EMA
            ema_fast = price_series.ewm(span=self.fast_period, adjust=False).mean()
            ema_slow = price_series.ewm(span=self.slow_period, adjust=False).mean()
            
            # MACD线和信号线
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
            histogram = macd_line - signal_line
            
            print(f"🔢 计算MACD: 数据范围 {min(prices):.2f} - {max(prices):.2f}")
            print(f"📐 MACD计算完成:")
            print(f"   EMA快线: {ema_fast.iloc[-1]:.2f}")
            print(f"   EMA慢线: {ema_slow.iloc[-1]:.2f}") 
            print(f"   MACD线: {macd_line.iloc[-1]:.6f}")
            print(f"   信号线: {signal_line.iloc[-1]:.6f}")
            print(f"   柱状图: {histogram.iloc[-1]:.6f}")
            
            return macd_line.tolist(), signal_line.tolist(), histogram.tolist()
            
        except Exception as e:
            print(f"❌ MACD计算错误: {e}")
            return [], [], []
    
    def _generate_signal(self, symbol: str, close_price: float, timestamp: float,
                        current_macd: float, current_signal: float, 
                        current_histogram: float, histogram: List[float]) -> Optional[TradingSignal]:
        """生成交易信号 - 改进的逻辑"""
        
        # 检查最小时间间隔（防止过度交易）
        if timestamp - self.last_signal_time[symbol] < self.min_trade_interval:
            print("💤 交易间隔太短，跳过信号")
            return None
        
        # 计算趋势强度
        trend_strength = self._calculate_trend_strength(symbol)
        
        # 改进的买入条件
        buy_signal, buy_reason = self._check_buy_condition(
            current_macd, current_signal, current_histogram, histogram, trend_strength
        )
        
        # 改进的卖出条件
        sell_signal, sell_reason = self._check_sell_condition(
            current_macd, current_signal, current_histogram, histogram, trend_strength
        )
        
        if buy_signal:
            return self._create_buy_signal(symbol, close_price, timestamp, current_histogram, histogram, buy_reason)
        elif sell_signal:
            return self._create_sell_signal(symbol, close_price, timestamp, current_histogram, histogram, sell_reason)
        
        print("💤 未满足信号条件")
        return None
    
def _check_buy_condition(self, current_macd: float, current_signal: float, 
                       current_histogram: float, histogram: List[float], 
                       trend_strength: float) -> Tuple[bool, str]:
    """紧急修复 - 更宽松的买入条件"""
    if len(histogram) < 2:
        return False, "数据不足"
    
    prev_histogram = histogram[-2]
    
    # 修复：移除对symbol的依赖，直接使用MACD历史数据
    macd_improving = True  # 默认认为在改善
    if len(histogram) >= 3:
        # 如果有足够的历史数据，检查MACD是否在改善
        prev_macd_diff = histogram[-3]  # 使用histogram作为代理
        macd_improving = current_histogram > prev_histogram
    
    hist_improving = current_histogram > prev_histogram
    
    # 多种买入情形 - 超级宽松的条件
    condition1 = (current_macd > current_signal)  # 简单金叉
    condition2 = (current_histogram > 0 and hist_improving)  # 正值改善
    condition3 = (prev_histogram <= 0 and current_histogram > 0)  # 负转正
    condition4 = (hist_improving and abs(current_histogram - prev_histogram) > 0)  # 任何改善
    
    # 调试输出条件状态
    print(f"🔍 买入条件检查:")
    print(f"   条件1(MACD金叉): {current_macd:.2f} > {current_signal:.2f} = {condition1}")
    print(f"   条件2(正值改善): {current_histogram:.2f} > 0 且 {current_histogram:.2f} > {prev_histogram:.2f} = {condition2}")
    print(f"   条件3(负转正): {prev_histogram:.2f} <= 0 且 {current_histogram:.2f} > 0 = {condition3}")
    print(f"   条件4(任何改善): {current_histogram:.2f} > {prev_histogram:.2f} = {condition4}")
    
    if condition1 or condition2 or condition3 or condition4:
        reason_parts = []
        if condition1: reason_parts.append("MACD金叉")
        if condition2: reason_parts.append("Hist正值改善") 
        if condition3: reason_parts.append("Hist负转正")
        if condition4: reason_parts.append("Hist改善")
        
        reason = f"买入: {', '.join(reason_parts)} ({prev_histogram:.4f}→{current_histogram:.4f})"
        print(f"🎯 满足买入条件: {reason}")
        return True, reason
    
    print("💤 所有买入条件都不满足")
    return False, "买入条件不满足"

    def _check_sell_condition(self, current_macd: float, current_signal: float,
                            current_histogram: float, histogram: List[float],
                            trend_strength: float) -> Tuple[bool, str]:
        """紧急修复 - 更宽松的卖出条件"""
        if len(histogram) < 2:
            return False, "数据不足"
        
        prev_histogram = histogram[-2]
        
        hist_worsening = current_histogram < prev_histogram
        
        # 多种卖出情形
        condition1 = (current_macd < current_signal)  # 简单死叉
        condition2 = (current_histogram < 0 and hist_worsening)  # 负值恶化
        condition3 = (prev_histogram >= 0 and current_histogram < 0)  # 正转负
        condition4 = (hist_worsening and current_histogram < 0)  # 负值继续恶化
        
        # 调试输出条件状态
        print(f"🔍 卖出条件检查:")
        print(f"   条件1(MACD死叉): {current_macd:.2f} < {current_signal:.2f} = {condition1}")
        print(f"   条件2(负值恶化): {current_histogram:.2f} < 0 且 {current_histogram:.2f} < {prev_histogram:.2f} = {condition2}")
        print(f"   条件3(正转负): {prev_histogram:.2f} >= 0 且 {current_histogram:.2f} < 0 = {condition3}")
        print(f"   条件4(负值恶化): {current_histogram:.2f} < {prev_histogram:.2f} 且 {current_histogram:.2f} < 0 = {condition4}")
        
        if condition1 or condition2 or condition3 or condition4:
            reason_parts = []
            if condition1: reason_parts.append("MACD死叉")
            if condition2: reason_parts.append("Hist负值恶化") 
            if condition3: reason_parts.append("Hist正转负")
            if condition4: reason_parts.append("Hist恶化")
            
            reason = f"卖出: {', '.join(reason_parts)} ({prev_histogram:.4f}→{current_histogram:.4f})"
            print(f"🎯 满足卖出条件: {reason}")
            return True, reason
        
        print("💤 所有卖出条件都不满足")
        return False, "卖出条件不满足"
    
    def _check_sell_condition(self, current_macd: float, current_signal: float,
                            current_histogram: float, histogram: List[float],
                            trend_strength: float) -> Tuple[bool, str]:
        """改进的卖出条件检查"""
        if len(histogram) < 3:
            return False, "数据不足"
        
        prev_histogram = histogram[-2]
        prev2_histogram = histogram[-3]
        
        # 条件1: MACD线下穿信号线或即将下穿
        macd_cross_down = current_macd < current_signal
        macd_near_cross_down = (current_macd >= current_signal and 
                               abs(current_macd - current_signal) < abs(current_macd) * 0.1)
        
        # 条件2: 柱状图恶化趋势
        hist_worsening = current_histogram < prev_histogram
        hist_negative_turn = prev_histogram >= 0 and current_histogram < 0
        hist_strong_decline = (prev_histogram > 0 and current_histogram < 0 and 
                              abs(current_histogram - prev_histogram) > self.hist_threshold)
        
        # 条件3: 柱状图连续恶化
        hist_trend_worsening = (current_histogram < prev_histogram < prev2_histogram)
        
        # 综合条件
        if ((macd_cross_down or macd_near_cross_down) and 
            (hist_strong_decline or (hist_worsening and hist_trend_worsening))):
            
            reason_parts = []
            if macd_cross_down:
                reason_parts.append("MACD死叉")
            elif macd_near_cross_down:
                reason_parts.append("MACD接近死叉")
                
            if hist_strong_decline:
                reason_parts.append(f"Hist强势转负({prev_histogram:.4f}→{current_histogram:.4f})")
            elif hist_worsening:
                reason_parts.append(f"Hist恶化趋势")
                
            reason = ", ".join(reason_parts)
            return True, reason
        
        return False, "卖出条件不满足"
    
    def _calculate_trend_strength(self, symbol: str) -> float:
        """计算价格趋势强度"""
        prices = list(self.price_data[symbol])
        if len(prices) < 10:
            return 0.0
        
        try:
            # 使用线性回归计算趋势
            x = np.arange(len(prices))
            y = np.array(prices)
            
            # 计算斜率作为趋势强度
            slope = np.polyfit(x, y, 1)[0]
            # 归一化到相对强度
            trend_strength = slope / np.mean(prices)
            return abs(trend_strength)
        except Exception as e:
            print(f"趋势强度计算错误: {e}")
            return 0.0
    
    def _create_buy_signal(self, symbol: str, price: float, timestamp: float,
                          current_histogram: float, histogram: List[float], reason: str) -> TradingSignal:
        """创建买入信号"""
        prev_hist = histogram[-2] if len(histogram) >= 2 else 0
        
        # 动态计算信号强度
        hist_change = abs(current_histogram - prev_hist)
        base_strength = min(hist_change * 1000, 0.8)
        trend_strength = self._calculate_trend_strength(symbol)
        final_strength = min(base_strength + trend_strength * 10, 0.95)
        
        # 确保最小强度
        final_strength = max(final_strength, 0.3)
        
        self.signal_count += 1
        self.last_signal_time[symbol] = timestamp
        
        print(f"🎯 {self.name} 买入信号 #{self.signal_count}")
        print(f"   原因: {reason}")
        print(f"   强度: {final_strength:.2f}")
        print(f"   价格: {price:.2f}")
        
        return TradingSignal(
            symbol=symbol,
            signal_type=SignalType.BUY,
            strength=final_strength,
            price=price,
            timestamp=timestamp,
            reason=reason,
            metadata={
                'histogram': current_histogram,
                'prev_histogram': prev_hist,
                'trend_strength': trend_strength
            }
        )
    
    def _create_sell_signal(self, symbol: str, price: float, timestamp: float,
                           current_histogram: float, histogram: List[float], reason: str) -> TradingSignal:
        """创建卖出信号"""
        prev_hist = histogram[-2] if len(histogram) >= 2 else 0
        
        hist_change = abs(current_histogram - prev_hist)
        base_strength = min(hist_change * 1000, 0.8)
        final_strength = min(base_strength, 0.95)
        final_strength = max(final_strength, 0.3)
        
        self.signal_count += 1
        self.last_signal_time[symbol] = timestamp
        
        print(f"🎯 {self.name} 卖出信号 #{self.signal_count}")
        print(f"   原因: {reason}")
        print(f"   强度: {final_strength:.2f}")
        print(f"   价格: {price:.2f}")
        
        return TradingSignal(
            symbol=symbol,
            signal_type=SignalType.SELL,
            strength=final_strength,
            price=price,
            timestamp=timestamp,
            reason=reason,
            metadata={
                'histogram': current_histogram,
                'prev_histogram': prev_hist
            }
        )
    
    def _print_debug_info(self, symbol: str, price: float, macd: float, 
                         signal: float, histogram: float, hist_list: List[float]):
        """打印调试信息"""
        print(f"📊 {self.name}:")
        print(f"   价格: {price:.2f}")
        print(f"   MACD: {macd:.6f}")
        print(f"   Signal: {signal:.6f}")
        print(f"   Histogram: {histogram:.6f}")
        
        if len(hist_list) >= 3:
            print(f"   Hist变化: {hist_list[-3]:.6f} -> {hist_list[-2]:.6f} -> {hist_list[-1]:.6f}")
        
        # 显示趋势强度
        trend_strength = self._calculate_trend_strength(symbol)
        print(f"   趋势强度: {trend_strength:.6f}")

# 测试函数
def create_trending_data():
    """创建有明显趋势的测试数据"""
    prices = []
    current = 50000
    
    # 先下跌趋势
    for i in range(20):
        current = current * (1 + np.random.normal(-0.002, 0.001))
        prices.append(current)
    
    # 然后上涨趋势
    for i in range(30):
        current = current * (1 + np.random.normal(0.0015, 0.001))
        prices.append(current)
    
    return prices

async def test_optimized_macd():
    """测试优化后的MACD策略"""
    print("🧪 测试优化版MACD策略...")
    print("=" * 60)
    
    strategy = MACDStrategyOptimized(
        name="MACD优化版",
        symbols=["BTC/USDT"],
        fast_period=12,
        slow_period=26,
        signal_period=9,
        min_trend_strength=0.0005,
        hist_threshold=0.00005,
        min_trade_interval=2
    )
    
    # 使用测试数据
    test_prices = create_trending_data()
    
    print(f"📊 测试数据: {len(test_prices)} 个价格点")
    print(f"📈 价格范围: {min(test_prices):.2f} - {max(test_prices):.2f}")
    
    # 创建市场数据对象
    class SimpleMarketData:
        def __init__(self, price, timestamp):
            self.symbol = "BTC/USDT"
            self.data = [timestamp, price, price+50, price-50, price, 1000]  # OHLCV格式
            self.timestamp = timestamp
    
    signals = []
    
    # 逐步喂数据，模拟实时交易
    for i, price in enumerate(test_prices):
        market_data = SimpleMarketData(price, i)
        signal = await strategy.analyze(market_data)
        
        if signal:
            signals.append(signal)
            print(f"✅ 捕获信号 #{len(signals)}: {signal.signal_type.value} @ {signal.price:.2f}")
            print(f"   原因: {signal.reason}")
            print(f"   强度: {signal.strength:.2f}")
    
    print(f"\n🎉 MACD优化版测试完成")
    print(f"📨 总生成信号: {len(signals)}")
    print(f"📊 测试数据趋势: 开始 {test_prices[0]:.2f} -> 结束 {test_prices[-1]:.2f}")
    
    if signals:
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        print(f"🛒 买入信号: {len(buy_signals)}")
        print(f"🏪 卖出信号: {len(sell_signals)}")
        
        # 简单策略评估
        if len(buy_signals) > 0 and len(sell_signals) > 0:
            first_buy = buy_signals[0].price
            last_sell = sell_signals[-1].price
            profit_pct = (last_sell - first_buy) / first_buy * 100
            print(f"💰 简单收益: {profit_pct:+.2f}%")
    else:
        print("❌ 未生成任何信号，需要进一步调试")

if __name__ == "__main__":
    asyncio.run(test_optimized_macd())