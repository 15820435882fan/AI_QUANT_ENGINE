# macd_strategy_optimized.py
#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
import asyncio
from collections import deque

class MACDStrategyOptimized(BaseStrategy):
    """MACD策略优化版 - 改进信号条件和风险管理"""
    
    def __init__(self, name: str, symbols: list, 
                 fast_period: int = 12, 
                 slow_period: int = 26, 
                 signal_period: int = 9,
                 min_trend_strength: float = 0.001,
                 hist_threshold: float = 0.0001):
        
        config = {
            'name': name,
            'symbols': symbols,
            'parameters': {
                'fast_period': fast_period,
                'slow_period': slow_period,
                'signal_period': signal_period
            }
        }
        super().__init__(config)
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.min_trend_strength = min_trend_strength
        self.hist_threshold = hist_threshold
        
        # 使用固定长度的数据队列
        self.price_data = {symbol: deque(maxlen=slow_period + signal_period + 20) for symbol in symbols}
        self.macd_history = {symbol: deque(maxlen=5) for symbol in symbols}  # 保存最近几个MACD值
        
        self.name = name
        self.signal_count = 0
        self.last_signal_time = {symbol: 0 for symbol in symbols}
    
    async def analyze(self, market_data) -> Optional[TradingSignal]:
        """优化后的MACD策略分析"""
        symbol = market_data.symbol
        
        # 提取收盘价
        close_price = self._extract_close_price(market_data)
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
        if not macd_line:
            return None
        
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        current_histogram = histogram[-1]
        
        # 更新MACD历史
        self.macd_history[symbol].append({
            'macd': current_macd,
            'signal': current_signal,
            'histogram': current_histogram,
            'price': close_price
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
            
            # 使用pandas计算
            price_series = pd.Series(prices)
            
            # 计算EMA
            ema_fast = price_series.ewm(span=self.fast_period, adjust=False).mean()
            ema_slow = price_series.ewm(span=self.slow_period, adjust=False).mean()
            
            # MACD线和信号线
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
            histogram = macd_line - signal_line
            
            return macd_line.tolist(), signal_line.tolist(), histogram.tolist()
            
        except Exception as e:
            print(f"❌ MACD计算错误: {e}")
            return [], [], []
    
    def _generate_signal(self, symbol: str, close_price: float, timestamp: float,
                        current_macd: float, current_signal: float, 
                        current_histogram: float, histogram: List[float]) -> Optional[TradingSignal]:
        """生成交易信号 - 改进的逻辑"""
        
        # 检查最小时间间隔（防止过度交易）
        if timestamp - self.last_signal_time[symbol] < 3:  # 至少3个时间单位
            return None
        
        # 计算趋势强度
        trend_strength = self._calculate_trend_strength(symbol)
        
        # 改进的买入条件
        buy_signal = self._check_buy_condition(current_macd, current_signal, current_histogram, histogram, trend_strength)
        
        # 改进的卖出条件
        sell_signal = self._check_sell_condition(current_macd, current_signal, current_histogram, histogram, trend_strength)
        
        if buy_signal:
            return self._create_buy_signal(symbol, close_price, timestamp, current_histogram, histogram)
        elif sell_signal:
            return self._create_sell_signal(symbol, close_price, timestamp, current_histogram, histogram)
        
        return None
    
    def _check_buy_condition(self, current_macd: float, current_signal: float, 
                           current_histogram: float, histogram: List[float], 
                           trend_strength: float) -> bool:
        """改进的买入条件检查"""
        if len(histogram) < 3:
            return False
        
        prev_histogram = histogram[-2]
        prev2_histogram = histogram[-3] if len(histogram) >= 3 else prev_histogram
        
        # 条件1: MACD线上穿信号线
        macd_cross_up = current_macd > current_signal
        
        # 条件2: 柱状图改善趋势（不要求严格从负转正）
        hist_improving = (
            (current_histogram > -self.hist_threshold and 
             current_histogram > prev_histogram) or  # 柱状图改善
            (prev_histogram <= 0 and current_histogram > 0)  # 或从负转正
        )
        
        # 条件3: 柱状图连续改善
        hist_trend = current_histogram > prev_histogram > prev2_histogram
        
        # 条件4: 有一定的趋势强度
        has_trend = trend_strength > self.min_trend_strength
        
        # 综合条件
        return (macd_cross_up and hist_improving and 
                (hist_trend or has_trend))
    
    def _check_sell_condition(self, current_macd: float, current_signal: float,
                            current_histogram: float, histogram: List[float],
                            trend_strength: float) -> bool:
        """改进的卖出条件检查"""
        if len(histogram) < 3:
            return False
        
        prev_histogram = histogram[-2]
        prev2_histogram = histogram[-3] if len(histogram) >= 3 else prev_histogram
        
        # 条件1: MACD线下穿信号线
        macd_cross_down = current_macd < current_signal
        
        # 条件2: 柱状图恶化趋势
        hist_worsening = (
            (current_histogram < self.hist_threshold and 
             current_histogram < prev_histogram) or  # 柱状图恶化
            (prev_histogram >= 0 and current_histogram < 0)  # 或从正转负
        )
        
        # 条件3: 柱状图连续恶化
        hist_trend = current_histogram < prev_histogram < prev2_histogram
        
        return macd_cross_down and hist_worsening and hist_trend
    
    def _calculate_trend_strength(self, symbol: str) -> float:
        """计算价格趋势强度"""
        prices = list(self.price_data[symbol])
        if len(prices) < 10:
            return 0.0
        
        # 使用线性回归计算趋势
        x = np.arange(len(prices))
        y = np.array(prices)
        
        try:
            # 计算斜率作为趋势强度
            slope = np.polyfit(x, y, 1)[0]
            # 归一化到相对强度
            trend_strength = slope / np.mean(prices)
            return abs(trend_strength)
        except:
            return 0.0
    
    def _create_buy_signal(self, symbol: str, price: float, timestamp: float,
                          current_histogram: float, histogram: List[float]) -> TradingSignal:
        """创建买入信号"""
        prev_hist = histogram[-2] if len(histogram) >= 2 else 0
        
        # 动态计算信号强度
        hist_change = abs(current_histogram - prev_hist)
        base_strength = min(hist_change * 1000, 0.8)  # 调整系数
        trend_strength = self._calculate_trend_strength(symbol)
        final_strength = min(base_strength + trend_strength * 10, 0.95)
        
        reason = f"MACD金叉, Hist改善: {prev_hist:.6f} -> {current_histogram:.6f}"
        
        self.signal_count += 1
        self.last_signal_time[symbol] = timestamp
        
        print(f"🎯 {self.name} 买入信号 #{self.signal_count}")
        print(f"   原因: {reason}")
        print(f"   强度: {final_strength:.2f}")
        
        return TradingSignal(
            symbol=symbol,
            signal_type=SignalType.BUY,
            strength=final_strength,
            price=price,
            timestamp=timestamp,
            reason=reason
        )
    
    def _create_sell_signal(self, symbol: str, price: float, timestamp: float,
                           current_histogram: float, histogram: List[float]) -> TradingSignal:
        """创建卖出信号"""
        prev_hist = histogram[-2] if len(histogram) >= 2 else 0
        
        hist_change = abs(current_histogram - prev_hist)
        base_strength = min(hist_change * 1000, 0.8)
        final_strength = min(base_strength, 0.95)
        
        reason = f"MACD死叉, Hist恶化: {prev_hist:.6f} -> {current_histogram:.6f}"
        
        self.signal_count += 1
        self.last_signal_time[symbol] = timestamp
        
        print(f"🎯 {self.name} 卖出信号 #{self.signal_count}")
        print(f"   原因: {reason}")
        print(f"   强度: {final_strength:.2f}")
        
        return TradingSignal(
            symbol=symbol,
            signal_type=SignalType.SELL,
            strength=final_strength,
            price=price,
            timestamp=timestamp,
            reason=reason
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
        min_trend_strength=0.0005,  # 降低趋势要求
        hist_threshold=0.00005      # 更敏感的柱状图阈值
    )
    
    # 使用相同的测试数据
    test_prices = create_trending_data()
    
    # ... 其余测试代码与之前相同