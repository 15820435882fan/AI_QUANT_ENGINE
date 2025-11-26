# multi_currency_mid_freq_system.py
#!/usr/bin/env python3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import asyncio

class MultiCurrencyMidFrequencySystem:
    """多币种中频交易系统 (5分钟级别，日交易5-10次)"""
    
    def __init__(self, currencies: List[str] = None):
        self.currencies = currencies or ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'DOT/USDT']
        self.timeframe = '5min'
        self.max_daily_trades = 10
        self.today_trades = {currency: 0 for currency in self.currencies}
        self.last_trade_date = datetime.now().date()
        
        # 策略权重
        self.strategy_weights = {
            'momentum': 0.3,
            'mean_reversion': 0.3,
            'breakout': 0.4
        }
    
    def reset_daily_counts(self):
        """重置每日交易计数"""
        today = datetime.now().date()
        if today != self.last_trade_date:
            self.today_trades = {currency: 0 for currency in self.currencies}
            self.last_trade_date = today
    
    async def analyze_currency(self, currency: str, market_data: pd.DataFrame) -> Dict:
        """分析单个币种"""
        self.reset_daily_counts()
        
        if self.today_trades[currency] >= self.max_daily_trades:
            return {'action': 'hold', 'reason': '达到日交易上限'}
        
        # 多策略分析
        momentum_signal = self.momentum_strategy(market_data)
        mean_reversion_signal = self.mean_reversion_strategy(market_data)
        breakout_signal = self.breakout_strategy(market_data)
        
        # 加权决策
        signals = {
            'buy': 0.0,
            'sell': 0.0,
            'hold': 0.0
        }
        
        for signal, weight in zip([momentum_signal, mean_reversion_signal, breakout_signal], 
                                self.strategy_weights.values()):
            signals[signal] += weight
        
        # 决定最终行动
        best_action = max(signals, key=signals.get)
        
        if best_action != 'hold':
            self.today_trades[currency] += 1
        
        return {
            'action': best_action,
            'confidence': signals[best_action],
            'today_trades': self.today_trades[currency],
            'reason': f"动量:{momentum_signal}, 均值回归:{mean_reversion_signal}, 突破:{breakout_signal}"
        }
    
    def momentum_strategy(self, data: pd.DataFrame) -> str:
        """动量策略"""
        if len(data) < 20:
            return 'hold'
        
        returns_5min = data['close'].pct_change(1).iloc[-1]
        returns_1h = (data['close'].iloc[-1] - data['close'].iloc[-12]) / data['close'].iloc[-12]
        
        if returns_5min > 0.002 and returns_1h > 0.005:  # 短期和中期动量
            return 'buy'
        elif returns_5min < -0.002 and returns_1h < -0.005:
            return 'sell'
        
        return 'hold'
    
    def mean_reversion_strategy(self, data: pd.DataFrame) -> str:
        """均值回归策略"""
        if len(data) < 50:
            return 'hold'
        
        current_price = data['close'].iloc[-1]
        ma_20 = data['close'].rolling(20).mean().iloc[-1]
        ma_50 = data['close'].rolling(50).mean().iloc[-1]
        
        deviation = (current_price - ma_20) / ma_20
        
        if deviation < -0.01:  # 价格低于均线1%
            return 'buy'
        elif deviation > 0.01:  # 价格高于均线1%
            return 'sell'
        
        return 'hold'
    
    def breakout_strategy(self, data: pd.DataFrame) -> str:
        """突破策略"""
        if len(data) < 20:
            return 'hold'
        
        current_high = data['high'].iloc[-1]
        resistance = data['high'].rolling(20).max().iloc[-2]  # 前20期最高点
        
        current_low = data['low'].iloc[-1]
        support = data['low'].rolling(20).min().iloc[-2]  # 前20期最低点
        
        if current_high > resistance * 1.001:  # 突破阻力
            return 'buy'
        elif current_low < support * 0.999:  # 跌破支撑
            return 'sell'
        
        return 'hold'
    
    async def run_daily_analysis(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """运行每日分析"""
        results = {}
        
        for currency in self.currencies:
            if currency in market_data:
                result = await self.analyze_currency(currency, market_data[currency])
                results[currency] = result
        
        return results

# 模拟测试
async def simulate_multi_currency_trading():
    """模拟多币种交易"""
    system = MultiCurrencyMidFrequencySystem()
    
    # 生成模拟数据 (1年，5分钟级别)
    currencies = system.currencies
    market_data = {}
    
    for currency in currencies:
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='5min')
        base_price = np.random.uniform(10, 50000)
        
        prices = [base_price]
        for i in range(1, len(dates)):
            # 模拟价格变动 (加密货币典型波动)
            volatility = 0.002  # 0.2% 5分钟波动
            change = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        market_data[currency] = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
            'close': prices
        })
    
    # 测试系统
    print("🚀 开始多币种中频交易系统测试")
    results = await system.run_daily_analysis(market_data)
    
    for currency, result in results.items():
        print(f"📊 {currency}: {result['action']} (置信度: {result['confidence']:.2f})")
        print(f"   今日交易: {result['today_trades']}/10, 原因: {result['reason']}")
    
    return results

if __name__ == "__main__":
    results = asyncio.run(simulate_multi_currency_trading())