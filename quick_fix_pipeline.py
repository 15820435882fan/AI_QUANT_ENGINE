# quick_fix_pipeline.py
"""
快速修复管道 - 使用已知工作的方法
"""
from real_market_data import RealMarketData
from enhanced_sniper_detector import EnhancedSniperDetector
from ai_strategy_optimizer_enhanced import EnhancedAIStrategyOptimizer

def quick_test():
    print("🎯 快速测试已知工作组件...")
    
    # 1. 数据层 - 使用健康检查中的方法
    market_data = RealMarketData()
    print("测试市场数据...")
    
    # 获取BTC数据 (参考健康检查)
    btc_data = market_data.get_recent_data('BTC/USDT', limit=50)
    print(f"BTC数据: {len(btc_data)}条")
    
    # 2. 信号层
    detector = EnhancedSniperDetector()
    print("信号检测器就绪")
    
    # 3. AI优化器
    optimizer = EnhancedAIStrategyOptimizer()
    print("AI优化器就绪")
    
    # 4. 直接运行生产系统测试
    from production_trading_system import ProductionTradingSystem
    production = ProductionTradingSystem()
    print("生产系统就绪")
    
    # 测试系统健康
    from system_health_check_final import run_health_check
    print("运行健康检查...")
    health_result = run_health_check()
    
    return {
        "data_working": len(btc_data) > 0,
        "signals_ready": detector is not None,
        "optimizer_ready": optimizer is not None, 
        "production_ready": production is not None,
        "health_check": "完成"
    }

if __name__ == "__main__":
    result = quick_test()
    print(f"\n📊 组件状态: {result}")