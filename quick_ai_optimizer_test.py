# quick_ai_optimizer_test.py
"""
快速测试并修复AI优化器，基于现有架构
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    from ai_strategy_optimizer_enhanced import AIStrategyOptimizer
    from real_market_data import RealMarketData
    
    # 1. 获取真实数据
    market_data = RealMarketData()
    data = market_data.get_historical_data('BTC/USDT', days=30, timeframe='1h')
    
    print(f"✅ 数据获取成功: {len(data)}条记录")
    print(f"📊 价格范围: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    # 2. 测试AI优化器
    optimizer = AIStrategyOptimizer()
    print("🚀 启动AI策略优化...")
    
    # 尝试优化SMA策略
    result = optimizer.optimize_strategy(
        strategy_class='SimpleMovingAverageStrategy',
        data=data,
        generations=5  # 快速测试
    )
    
    print(f"🎯 AI优化结果: {result}")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    print("🔧 需要修复AI优化器...") 