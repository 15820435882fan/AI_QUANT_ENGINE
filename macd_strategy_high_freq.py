# macd_strategy_high_freq.py
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

class MACDStrategyHighFrequency:
    """高频MACD策略 - 增加交易机会，降低门槛"""
    
    def __init__(self, name: str, symbols: List[str], 
                 fast_period: int = 8,      # 更快的参数
                 slow_period: int = 21,     # 稍短的慢线
                 signal_period: int = 5,    # 更快的信号线
                 min_trade_interval: int = 2,
                 profit_target: float = 0.05,  # 5%止盈
                 stop_loss: float = 0.02):     # 2%止损
        
        self.name = name
        self.symbols = symbols
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.min_trade_interval = min_trade_interval
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        
        # 数据存储
        self.price_data = {symbol: deque(maxlen=100) for symbol in symbols}
        self.signal_count = 0
        self.last_signal_time = {symbol: 0 for symbol in symbols}
        self.current_position = {symbol: None for symbol in symbols}
        self.entry_price = {symbol: 0 for symbol in symbols}  # 入场价格
    
    async def analyze(self, market_data) -> Optional[TradingSignal]:
        """高频MACD分析"""
        symbol = market_data.symbol
        
        # 提取收盘价
        close_price = self._extract_close_price(market_data)
        if close_price is None:
            return None
        
        # 更新价格数据
        self.price_data[symbol].append(close_price)
        
        # 检查最小数据长度
        min_data_length = max(self.slow_period + self.signal_period, 10)
        if len(self.price_data[symbol]) < min_data_length:
            return None
        
        # 检查交易间隔
        if market_data.timestamp - self.last_signal_time[symbol] < self.min_trade_interval:
            # 但检查止损止盈
            stop_signal = self._check_stop_loss_take_profit(symbol, close_price, market_data.timestamp)
            if stop_signal:
                return stop_signal
            return None
        
        # 计算MACD指标
        macd_line, signal_line, histogram = self._calculate_macd(symbol)
        if not macd_line or len(macd_line) == 0:
            return None
        
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        current_histogram = histogram[-1]
        
        # 高频信号生成逻辑
        signal = self._generate_high_freq_signal(
            symbol, close_price, market_data.timestamp,
            current_macd, current_signal, current_histogram, histogram
        )
        
        return signal
    
    def _generate_high_freq_signal(self, symbol: str, close_price: float, timestamp: float,
                                 current_macd: float, current_signal: float, 
                                 current_histogram: float, histogram: List[float]) -> Optional[TradingSignal]:
        """高频信号生成逻辑 - 降低门槛"""
        
        if len(histogram) < 2:
            return None
        
        prev_histogram = histogram[-2]
        
        # 更宽松的买入条件
        if self.current_position[symbol] in [None, 'short']:
            # 条件1: 简单金叉
            if current_macd > current_signal and prev_histogram <= current_macd:
                reason = f"高频金叉 (MACD: {prev_histogram:.1f}→{current_macd:.1f})"
                self.current_position[symbol] = 'long'
                self.entry_price[symbol] = close_price
                return self._create_signal(symbol, close_price, timestamp, SignalType.BUY, reason, 0.6)
            
            # 条件2: 柱状图转正
            if prev_histogram < 0 and current_histogram > 0:
                reason = f"柱状图转正 ({prev_histogram:.1f}→{current_histogram:.1f})"
                self.current_position[symbol] = 'long'
                self.entry_price[symbol] = close_price
                return self._create_signal(symbol, close_price, timestamp, SignalType.BUY, reason, 0.7)
            
            # 条件3: 柱状图大幅改善
            if current_histogram > prev_histogram and (current_histogram - prev_histogram) > abs(prev_histogram) * 0.5:
                reason = f"柱状图大幅改善 ({prev_histogram:.1f}→{current_histogram:.1f})"
                self.current_position[symbol] = 'long'
                self.entry_price[symbol] = close_price
                return self._create_signal(symbol, close_price, timestamp, SignalType.BUY, reason, 0.5)
        
        # 更积极的卖出条件
        if self.current_position[symbol] == 'long':
            # 条件1: 简单死叉
            if current_macd < current_signal and prev_histogram >= current_macd:
                reason = f"高频死叉 (MACD: {prev_histogram:.1f}→{current_macd:.1f})"
                self.current_position[symbol] = None
                return self._create_signal(symbol, close_price, timestamp, SignalType.SELL, reason, 0.6)
            
            # 条件2: 柱状图转负
            if prev_histogram > 0 and current_histogram < 0:
                reason = f"柱状图转负 ({prev_histogram:.1f}→{current_histogram:.1f})"
                self.current_position[symbol] = None
                return self._create_signal(symbol, close_price, timestamp, SignalType.SELL, reason, 0.7)
            
            # 条件3: 柱状图大幅恶化
            if current_histogram < prev_histogram and (prev_histogram - current_histogram) > abs(prev_histogram) * 0.3:
                reason = f"柱状图大幅恶化 ({prev_histogram:.1f}→{current_histogram:.1f})"
                self.current_position[symbol] = None
                return self._create_signal(symbol, close_price, timestamp, SignalType.SELL, reason, 0.5)
        
        return None
    
    def _check_stop_loss_take_profit(self, symbol: str, current_price: float, timestamp: float) -> Optional[TradingSignal]:
        """检查止损止盈"""
        if self.current_position[symbol] == 'long' and self.entry_price[symbol] > 0:
            profit_pct = (current_price - self.entry_price[symbol]) / self.entry_price[symbol]
            
            # 止盈
            if profit_pct >= self.profit_target:
                reason = f"达到止盈目标 (+{profit_pct*100:.1f}%)"
                self.current_position[symbol] = None
                return self._create_signal(symbol, current_price, timestamp, SignalType.SELL, reason, 0.8)
            
            # 止损
            if profit_pct <= -self.stop_loss:
                reason = f"触发止损 ({profit_pct*100:.1f}%)"
                self.current_position[symbol] = None
                return self._create_signal(symbol, current_price, timestamp, SignalType.SELL, reason, 0.9)
        
        return None
    
    def _calculate_macd(self, symbol: str) -> Tuple[List[float], List[float], List[float]]:
        """计算MACD指标"""
        try:
            prices = list(self.price_data[symbol])
            
            if len(prices) < self.slow_period:
                return [], [], []
            
            price_series = pd.Series(prices)
            ema_fast = price_series.ewm(span=self.fast_period, adjust=False).mean()
            ema_slow = price_series.ewm(span=self.slow_period, adjust=False).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
            histogram = macd_line - signal_line
            
            return macd_line.tolist(), signal_line.tolist(), histogram.tolist()
            
        except Exception as e:
            print(f"❌ MACD计算错误: {e}")
            return [], [], []
    
    def _create_signal(self, symbol: str, price: float, timestamp: float, 
                      signal_type: SignalType, reason: str, strength: float) -> TradingSignal:
        """创建交易信号"""
        self.signal_count += 1
        self.last_signal_time[symbol] = timestamp
        
        print(f"🎯 {self.name} 信号 #{self.signal_count}: {signal_type.value}")
        print(f"   原因: {reason}")
        print(f"   价格: {price:.2f}")
        
        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            price=price,
            timestamp=timestamp,
            reason=reason
        )
    
    def _extract_close_price(self, market_data):
        """提取收盘价"""
        try:
            if hasattr(market_data, 'data') and isinstance(market_data.data, (list, tuple)) and len(market_data.data) >= 5:
                return float(market_data.data[4])
            return None
        except:
            return None