# test_adaptive_system.py
#!/usr/bin/env python3
import sys
import os
import asyncio
import logging
import pandas as pd
import numpy as np  # 🔧 添加这行

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.strategies.market_regime_detector import MarketRegimeDetector
from src.strategies.multi_strategy_manager import MultiStrategyManager

async def test_adaptive_system():
    """测试自适应交易系统"""
    print("🧪 测试自适应交易系统...")
    
    # 创建组件
    regime_detector = MarketRegimeDetector()
    strategy_manager = MultiStrategyManager()
    
    # 生成测试数据
    data = generate_test_data()
    print(f"📊 生成 {len(data)} 条测试数据")
    
    # 测试市场状态检测
    regime = await regime_detector.detect_regime(data)
    confidence = await regime_detector.get_regime_confidence(data)
    
    print(f"🎯 检测到的市场状态: {regime}")
    print(f"📊 状态置信度:")
    for reg, conf in confidence.items():
        print(f"  {reg}: {conf:.2%}")
    
    # 测试策略选择
    await strategy_manager.update_market_regime(data)
    active_strategies = strategy_manager.get_active_strategies()
    
    print(f"\n🚀 激活的策略:")
    for strategy in active_strategies:
        print(f"  📈 {strategy['name']} ({strategy['type']})")
    
    print("\n✅ 自适应系统测试完成")

def generate_test_data(days: int = 30) -> pd.DataFrame:
    """生成测试数据"""
    dates = pd.date_range(start="2024-01-01", periods=days*1440, freq='1min')
    
    # 模拟不同市场状态的数据
    data = []
    price = 50000.0
    trend_direction = 1  # 1:上涨, -1:下跌
    
    for i, date in enumerate(dates):
        # 模拟市场状态变化
        if i % 10000 == 0:  # 每10000个点改变趋势
            trend_direction *= -1
        
        # 价格波动
        trend_component = trend_direction * 0.0001  # 趋势成分
        noise = np.random.normal(0, 0.001)  # 噪声成分
        
        price = price * (1 + trend_component + noise)
        price = max(price, 1000)  # 防止价格归零
        
        data.append({
            'timestamp': date,
            'open': price * (1 + np.random.normal(0, 0.0005)),
            'high': price * (1 + abs(np.random.normal(0, 0.001))),
            'low': price * (1 - abs(np.random.normal(0, 0.001))),
            'close': price,
            'volume': np.random.uniform(1000, 5000)
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_adaptive_system())