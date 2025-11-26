# test_trading_simple.py
#!/usr/bin/env python3
import sys
import os
import asyncio
import logging

# 设置路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_trading_simple():
    """简化版交易系统测试"""
    print("🧪 运行简化版交易测试...")
    
    try:
        # 测试核心组件
        from src.risk.risk_manager import RiskManager, RiskConfig
        from src.backtesting.backtest_engine import DataManager
        
        # 1. 测试数据加载
        print("📊 测试数据加载...")
        data_manager = DataManager()
        data = await data_manager.load_historical_data("BTC/USDT", "2024-01-01", "2024-01-02")
        print(f"✅ 数据加载: {len(data)} 条记录")
        
        # 2. 测试风险管理
        print("🛡️ 测试风险管理...")
        risk_manager = RiskManager(RiskConfig())
        signal = {'action': 'buy', 'price': 50000.0}
        result = await risk_manager.validate_trade(signal, 10000.0, {}, 0)
        print(f"✅ 风险检查: {result['reason']}")
        
        # 3. 测试策略组件
        print("🎯 测试策略组件...")
        from src.strategies.market_regime_detector import MarketRegimeDetector
        detector = MarketRegimeDetector()
        regime = await detector.detect_regime(data)
        print(f"✅ 市场检测: {regime}")
        
        print("\n🎉 交易系统核心组件测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_trading_simple())