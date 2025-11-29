# working_quant_pipeline.py
"""
工作版量化流水线 - 使用正确的类名和方法名
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

def run_working_pipeline():
    print("🚀 启动工作版量化流水线...")
    
    try:
        # 1. 数据层 - 使用正确的方法名
        from real_market_data import RealMarketData
        market_data = RealMarketData()
        
        # 查看RealMarketData的实际方法
        print("📋 RealMarketData的方法:", [m for m in dir(market_data) if not m.startswith('_')])
        
        # 尝试不同的数据获取方法
        if hasattr(market_data, 'fetch_market_data'):
            data = market_data.fetch_market_data('BTC/USDT', days=7)
        elif hasattr(market_data, 'get_data'):
            data = market_data.get_data('BTC/USDT', days=7) 
        else:
            # 使用健康检查中的方法
            data = market_data.get_recent_data('BTC/USDT', limit=100)
        
        print(f"✅ 数据获取: {len(data) if data is not None else '无'}条记录")
        
        # 2. 信号层
        from enhanced_sniper_detector import EnhancedSniperDetector
        detector = EnhancedSniperDetector()
        
        # 查看信号检测器方法
        print("📋 EnhancedSniperDetector方法:", [m for m in dir(detector) if not m.startswith('_')])
        
        if hasattr(detector, 'analyze_enhanced_signals'):
            signals = detector.analyze_enhanced_signals(data, 'BTC/USDT')
        else:
            signals = detector.generate_signals(data)
            
        print(f"✅ 信号生成: {len(signals) if signals else 0}个信号")
        
        # 3. AI优化层 - 使用正确的类名
        from ai_strategy_optimizer_enhanced import EnhancedAIStrategyOptimizer
        optimizer = EnhancedAIStrategyOptimizer()
        print("✅ AI优化器初始化完成")
        
        # 查看优化器方法
        print("📋 EnhancedAIStrategyOptimizer方法:", [m for m in dir(optimizer) if not m.startswith('_')])
        
        # 4. 运行优化
        if hasattr(optimizer, 'optimize_strategy'):
            result = optimizer.optimize_strategy(
                strategy_class='SimpleMovingAverageStrategy',
                data=data,
                generations=3
            )
            print(f"🎯 AI优化结果: {result}")
        else:
            print("⚠️ 优化器没有optimize_strategy方法")
            
        # 5. 回测验证
        from high_frequency_backtest import HighFrequencyBacktest
        backtester = HighFrequencyBacktest()
        print("✅ 回测系统就绪")
        
        return "🚀 流水线执行完成"
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return f"错误: {e}"

if __name__ == "__main__":
    result = run_working_pipeline()
    print(f"\n🎉 最终结果: {result}")