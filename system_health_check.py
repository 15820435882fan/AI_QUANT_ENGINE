# system_health_check.py
#!/usr/bin/env python3
import sys
import os
import asyncio
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def check_data_pipeline():
    """检查数据管道"""
    try:
        from src.data.data_pipeline import DataPipeline
        pipeline = DataPipeline(symbols=["BTC/USDT"])
        await pipeline.initialize()
        await pipeline.stop()
        return True
    except Exception as e:
        print(f"❌ 数据管道错误: {e}")
        return False

async def check_strategies():
    """检查策略系统"""
    try:
        from src.strategies.multi_strategy_manager import MultiStrategyManager
        from src.strategies.market_regime_detector import MarketRegimeDetector
        
        manager = MultiStrategyManager()
        detector = MarketRegimeDetector()
        
        # 生成测试数据
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range(start="2024-01-01", periods=100, freq='1min')
        data = []
        price = 50000.0
        
        for date in dates:
            change = np.random.normal(0, 0.001)
            price = price * (1 + change)
            data.append({
                'timestamp': date,
                'open': price, 'high': price*1.001, 'low': price*0.999, 'close': price,
                'volume': np.random.uniform(1000, 5000)
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # 测试市场检测
        regime = await detector.detect_regime(df)
        print(f"📊 市场状态检测: {regime}")
        
        # 测试策略选择
        await manager.update_market_regime(df)
        strategies = manager.get_active_strategies()
        print(f"🎯 策略选择: {len(strategies)}个策略")
        
        return True
    except Exception as e:
        print(f"❌ 策略系统错误: {e}")
        return False

async def check_risk_management():
    """检查风险管理系统"""
    try:
        from src.risk.risk_manager import RiskManager, RiskConfig
        
        risk_manager = RiskManager(RiskConfig())
        
        # 测试交易验证
        test_signal = {
            'action': 'buy',
            'price': 50000.0
        }
        
        risk_result = await risk_manager.validate_trade(
            test_signal, 
            current_equity=10000.0,
            positions={},
            today_trades=5
        )
        
        print(f"🛡️ 风险检查: {risk_result['approved']} - {risk_result['reason']}")
        return True
    except Exception as e:
        print(f"❌ 风险管理系统错误: {e}")
        return False

async def check_order_execution():
    """检查订单执行"""
    try:
        from src.trading.live_trader import LiveTrader
        
        trader = LiveTrader(paper_trading=True)
        
        # 测试短时间运行
        print("🔧 测试交易引擎...")
        await asyncio.wait_for(trader.start_trading(), timeout=10)
        await trader.stop_trading()
        
        return True
    except asyncio.TimeoutError:
        print("✅ 交易引擎正常超时")
        return True
    except Exception as e:
        print(f"❌ 订单执行错误: {e}")
        return False

async def check_backtest_system():
    """检查回测系统"""
    try:
        from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig, DataManager
        from src.backtesting.backtest_strategies import RobustSMAStrategy
        
        config = BacktestConfig(initial_capital=10000.0)
        engine = BacktestEngine(config)
        data_manager = DataManager()
        
        historical_data = await data_manager.load_historical_data(
            "BTC/USDT", "2024-01-01", "2024-01-05"
        )
        
        strategy = RobustSMAStrategy(
            name="健康检查策略", 
            symbols=["BTC/USDT"],
            fast_period=5,
            slow_period=15
        )
        
        result = await engine.run_backtest(strategy, historical_data)
        print(f"📈 回测系统: {result.total_return:.2%} 收益")
        
        return True
    except Exception as e:
        print(f"❌ 回测系统错误: {e}")
        return False

async def health_check():
    """系统健康检查"""
    print("🧪 开始系统健康检查...")
    print("=" * 50)
    
    checks = {
        '数据管道': await check_data_pipeline(),
        '策略系统': await check_strategies(), 
        '风险管理': await check_risk_management(),
        '交易执行': await check_order_execution(),
        '回测系统': await check_backtest_system()
    }
    
    print("\n" + "=" * 50)
    print("📊 系统健康报告:")
    for component, status in checks.items():
        print(f"  {component}: {'✅' if status else '❌'}")
    
    overall_status = all(checks.values())
    print(f"\n🏆 总体状态: {'✅ 健康' if overall_status else '❌ 需要修复'}")
    
    return overall_status

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(health_check())