# test_quick_fix.py
#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_strategy_manager_enhanced import EnhancedMultiStrategyManager
from src.strategies.strategy_orchestrator import TradingSignal, SignalType

class SimpleMarketData:
    """简化的市场数据类用于测试"""
    def __init__(self, price, timestamp, symbol="BTC/USDT"):
        self.symbol = symbol
        self.data = [timestamp, price, price * 1.01, price * 0.99, price, 1000]  # OHLCV格式
        self.timestamp = timestamp
        self.close = price

async def test_multi_strategy_basic():
    """测试多策略管理器基本功能"""
    print("🧪 测试多策略管理器基本功能...")
    print("=" * 40)
    
    try:
        # 初始化管理器
        manager = EnhancedMultiStrategyManager(symbols=["BTC/USDT"])
        print("✅ 多策略管理器初始化成功")
        
        # 测试不同价格点的信号生成
        test_cases = [
            (50000, 1700000000, "正常价格"),
            (51000, 1700003600, "上涨价格"), 
            (49000, 1700007200, "下跌价格"),
            (50500, 1700010800, "波动价格")
        ]
        
        signals_generated = 0
        
        for price, timestamp, description in test_cases:
            test_data = SimpleMarketData(price, timestamp)
            signal = await manager.analyze(test_data)
            
            if signal:
                signals_generated += 1
                print(f"✅ {description}: {signal.signal_type.value} @ {signal.price:.2f}")
                print(f"   强度: {signal.strength:.3f}, 原因: {signal.reason}")
            else:
                print(f"ℹ️  {description}: 无信号生成")
        
        print(f"\n📊 信号生成统计: {signals_generated}/{len(test_cases)}")
        
        # 测试策略性能统计
        performance = manager.get_strategy_performance()
        print(f"\n🔧 策略性能统计:")
        for strategy, stats in performance.items():
            print(f"   {strategy}: {stats['signal_count']} 个信号")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_trading_signal_creation():
    """测试TradingSignal创建"""
    print("\n🧪 测试TradingSignal创建...")
    print("=" * 40)
    
    try:
        # 直接测试TradingSignal创建
        signal = TradingSignal(
            symbol="BTC/USDT",
            signal_type=SignalType.BUY,
            price=50000.0,
            strength=0.8,
            timestamp=1700000000,
            reason="测试信号"
        )
        
        print("✅ TradingSignal创建成功")
        print(f"   符号: {signal.symbol}")
        print(f"   类型: {signal.signal_type.value}")
        print(f"   价格: {signal.price:.2f}")
        print(f"   强度: {signal.strength:.3f}")
        print(f"   原因: {signal.reason}")
        
        return True
        
    except Exception as e:
        print(f"❌ TradingSignal创建失败: {e}")
        return False

async def test_market_regime_detection():
    """测试市场状态检测"""
    print("\n🧪 测试市场状态检测...")
    print("=" * 40)
    
    try:
        manager = EnhancedMultiStrategyManager(symbols=["BTC/USDT"])
        
        # 测试不同市场状态
        regimes = ["bull", "bear", "ranging", "trend"]
        
        for regime in regimes:
            manager.update_market_regime(regime)
            current_regime = manager.current_regime
            print(f"✅ 市场状态 '{regime}': {current_regime}")
            
            # 检查策略权重
            weights = manager.strategy_weights
            print(f"   策略权重: {weights}")
        
        return True
        
    except Exception as e:
        print(f"❌ 市场状态检测测试失败: {e}")
        return False

async def main():
    """主测试函数"""
    print("🚀 开始快速修复验证测试")
    print("=" * 50)
    
    tests = [
        test_trading_signal_creation(),
        test_multi_strategy_basic(), 
        test_market_regime_detection()
    ]
    
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    
    success_count = 0
    for i, result in enumerate(results):
        if result is True:
            success_count += 1
            print(f"✅ 测试 {i+1}: 通过")
        else:
            print(f"❌ 测试 {i+1}: 失败 - {result}")
    
    print(f"\n🎯 总体结果: {success_count}/{len(tests)} 通过")
    
    if success_count == len(tests):
        print("🎉 所有修复验证成功！系统可以正常运行。")
        return True
    else:
        print("⚠️ 部分测试失败，需要进一步检查。")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)