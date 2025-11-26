# aggressive_multi_currency.py
#!/usr/bin/env python3
from typing import List, Dict, Any, Optional  # 添加这行导入
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio

class AggressiveMultiCurrencySystem:
    """激进版多币种交易系统 - 提高信号频率"""
    
    def __init__(self, currencies: List[str] = None):
        self.currencies = currencies or ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'DOT/USDT']
        self.max_daily_trades = 15  # 提高交易上限
        
        # 更激进的策略权重
        self.strategy_weights = {
            'momentum': 0.25,
            'reversion': 0.35, 
            'breakout': 0.40
        }
    
    async def analyze_currency(self, currency: str, data: pd.DataFrame) -> Dict:
        """激进版分析"""
        if len(data) < 50:
            return {'action': 'hold', 'reason': '数据不足'}
        
        # 多策略信号
        signals = {
            'momentum': self.aggressive_momentum(data),
            'reversion': self.aggressive_reversion(data),
            'breakout': self.aggressive_breakout(data)
        }
        
        # 计算加权得分
        buy_score = 0
        sell_score = 0
        
        for strategy, signal in signals.items():
            weight = self.strategy_weights[strategy]
            if signal == 'buy':
                buy_score += weight
            elif signal == 'sell':
                sell_score += weight
        
        # 降低触发阈值
        if buy_score > 0.4:  # 从0.7降低到0.4
            action = 'buy'
            confidence = buy_score
        elif sell_score > 0.4:
            action = 'sell' 
            confidence = sell_score
        else:
            action = 'hold'
            confidence = max(buy_score, sell_score)
        
        return {
            'action': action,
            'confidence': confidence,
            'signals': signals,
            'reason': f"动量:{signals['momentum']}, 回归:{signals['reversion']}, 突破:{signals['breakout']}"
        }
    
    def aggressive_momentum(self, data: pd.DataFrame) -> str:
        """激进动量策略"""
        current_price = data['close'].iloc[-1]
        
        # 短期动量 (5期)
        returns_5 = (current_price - data['close'].iloc[-5]) / data['close'].iloc[-5]
        # 中期动量 (20期)  
        returns_20 = (current_price - data['close'].iloc[-20]) / data['close'].iloc[-20]
        
        if returns_5 > 0.005 or returns_20 > 0.01:  # 降低阈值
            return 'buy'
        elif returns_5 < -0.005 or returns_20 < -0.01:
            return 'sell'
        
        return 'hold'
    
    def aggressive_reversion(self, data: pd.DataFrame) -> str:
        """激进均值回归策略"""
        current_price = data['close'].iloc[-1]
        ma_10 = data['close'].rolling(10).mean().iloc[-1]
        ma_30 = data['close'].rolling(30).mean().iloc[-1]
        
        # 计算与均线的偏离
        dev_10 = (current_price - ma_10) / ma_10
        dev_30 = (current_price - ma_30) / ma_30
        
        if dev_10 < -0.008 or dev_30 < -0.015:  # 降低超卖阈值
            return 'buy'
        elif dev_10 > 0.008 or dev_30 > 0.015:   # 降低超买阈值
            return 'sell'
        
        return 'hold'
    
    def aggressive_breakout(self, data: pd.DataFrame) -> str:
        """激进突破策略"""
        current_high = data['high'].iloc[-1]
        current_low = data['low'].iloc[-1]
        
        # 近期高低点
        resistance_10 = data['high'].rolling(10).max().iloc[-2]
        support_10 = data['low'].rolling(10).min().iloc[-2]
        
        resistance_20 = data['high'].rolling(20).max().iloc[-2] 
        support_20 = data['low'].rolling(20).min().iloc[-2]
        
        # 突破检测
        if current_high > resistance_10 * 1.0005 or current_high > resistance_20 * 1.001:
            return 'buy'
        elif current_low < support_10 * 0.9995 or current_low < support_20 * 0.999:
            return 'sell'
        
        return 'hold'

async def test_aggressive_system():
    """测试激进系统"""
    system = AggressiveMultiCurrencySystem()
    
    # 生成更有波动的测试数据
    results = {}
    
    for currency in system.currencies:
        # 创建波动数据
        prices = [100]
        for i in range(200):
            # 更大的波动
            change = np.random.normal(0, 0.01)  # 1%波动
            trend = 0.0002 * i  # 轻微趋势
            new_price = prices[-1] * (1 + change + trend)
            prices.append(max(new_price, 1))
        
        data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.008 for p in prices],
            'low': [p * 0.992 for p in prices], 
            'close': prices
        })
        
        result = await system.analyze_currency(currency, data)
        results[currency] = result
        
        print(f"📊 {currency}: {result['action']} (置信度: {result['confidence']:.2f})")
        print(f"   信号详情: {result['reason']}")
    
    # 统计结果
    actions = [r['action'] for r in results.values()]
    buy_count = actions.count('buy')
    sell_count = actions.count('sell')
    
    print(f"\n📈 总体统计: 买入 {buy_count}, 卖出 {sell_count}, 观望 {len(actions)-buy_count-sell_count}")
    
    return results

if __name__ == "__main__":
    print("🚀 测试激进版多币种交易系统")
    results = asyncio.run(test_aggressive_system())