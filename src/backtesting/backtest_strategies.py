# src/backtesting/backtest_strategies.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测专用策略 - 完全兼容回测数据格式
"""

import sys
import os
import pandas as pd
from typing import Optional

# 🔧 修复导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.strategies.strategy_orchestrator import BaseStrategy, TradingSignal, SignalType
from src.data.data_pipeline import DataType

class RobustSMAStrategy(BaseStrategy):
    """鲁棒的SMA策略 - 专门用于回测"""
    
    def __init__(self, name: str, symbols: list, fast_period: int = 5, slow_period: int = 10):
        config = {
            'name': name,
            'symbols': symbols,
            'parameters': {
                'fast_period': fast_period,
                'slow_period': slow_period
            }
        }
        super().__init__(config)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.data_buffer = {symbol: [] for symbol in symbols}
    
    async def analyze(self, market_data) -> Optional[TradingSignal]:
        """分析市场数据 - 完全兼容各种数据格式"""
        # 只处理OHLCV数据
        if hasattr(market_data, 'data_type') and market_data.data_type != DataType.OHLCV:
            return None
            
        symbol = market_data.symbol
        
        # 获取收盘价 - 兼容所有数据格式
        close_price = self._extract_close_price(market_data)
        if close_price is None:
            return None
        
        # 添加到数据缓冲区
        if symbol not in self.data_buffer:
            self.data_buffer[symbol] = []
        
        buffer = self.data_buffer[symbol]
        buffer.append(close_price)
        
        # 保持缓冲区大小
        if len(buffer) > self.slow_period:
            buffer.pop(0)
        
        # 检查是否有足够数据
        min_data_required = min(self.fast_period, 3)
        if len(buffer) < min_data_required:
            print(f"⏳ 策略数据收集中... ({len(buffer)}/{min_data_required})")
            return None
        
        # 计算移动平均
        actual_fast_period = min(self.fast_period, len(buffer))
        actual_slow_period = min(self.slow_period, len(buffer))
        
        fast_ma = sum(buffer[-actual_fast_period:]) / actual_fast_period
        slow_ma = sum(buffer[-actual_slow_period:]) / actual_slow_period
        
        current_price = buffer[-1]
        
        print(f"📊 SMA策略分析: {symbol} 快速MA={fast_ma:.2f}, 慢速MA={slow_ma:.2f}, 当前价={current_price:.2f}")
        
        # 生成交易信号
        signal_strength = abs(fast_ma - slow_ma) / current_price
        
        if fast_ma > slow_ma * 1.001:
            print(f"🎯 生成买入信号! 快速MA({fast_ma:.2f}) > 慢速MA({slow_ma:.2f})")
            return TradingSignal(
                symbol=market_data.symbol,
                action="BUY",
                confidence=confidence,
                timestamp=market_data.timestamp
            )
        elif fast_ma < slow_ma * 0.999:
            print(f"🎯 生成卖出信号! 快速MA({fast_ma:.2f}) < 慢速MA({slow_ma:.2f})")
            return TradingSignal(
                symbol=market_data.symbol,
                action="SELL",
                confidence=confidence,
                timestamp=market_data.timestamp
        )
        
        return None
    
    def _extract_close_price(self, market_data):
        """从市场数据中提取收盘价 - 兼容所有格式"""
        data = market_data.data
        
        try:
            if isinstance(data, (list, tuple)) and len(data) >= 5:
                # 列表格式: [timestamp, open, high, low, close, volume]
                return float(data[4])
            elif isinstance(data, dict):
                # 字典格式
                if 'close' in data:
                    return float(data['close'])
                elif 'last' in data:
                    return float(data['last'])
            elif isinstance(data, (int, float)):
                # 直接是价格
                return float(data)
        except (ValueError, TypeError, IndexError) as e:
            print(f"⚠️ 无法提取收盘价: {e}")
        
        return None

# 测试策略
async def test_robust_strategy():
    """测试鲁棒策略"""
    print("🧪 测试鲁棒策略...")
    
    strategy = RobustSMAStrategy(
        name="测试策略",
        symbols=["BTC/USDT"],
        fast_period=3,
        slow_period=5
    )
    
    # 创建测试数据
    from src.data.data_pipeline import MarketData, DataType
    import time
    
    test_data = MarketData(
        symbol="BTC/USDT",
        data_type=DataType.OHLCV,
        data=[time.time(), 50000, 51000, 49000, 50500, 1000],  # 列表格式
        timestamp=time.time()
    )
    
    signal = await strategy.analyze(test_data)
    if signal:
        print(f"✅ 策略生成信号: {signal.signal_type.value}")
    else:
        print("ℹ️ 策略未生成信号（可能需要更多数据）")
    
    print("✅ 鲁棒策略测试完成")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_robust_strategy())