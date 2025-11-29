# health_check_based_pipeline.py
"""
基于健康检查模式的量化流水线
"""
import logging
logging.basicConfig(level=logging.INFO)

def health_check_based_test():
    print("🎯 基于健康检查模式测试...")
    
    # 直接复制健康检查中工作的代码模式
    from real_market_data import RealMarketData
    from enhanced_sniper_detector import EnhancedSniperDetector
    from production_trading_system import ProductionTradingSystem
    
    # 初始化组件（参考健康检查）
    market_data = RealMarketData()
    detector = EnhancedSniperDetector()
    production = ProductionTradingSystem()
    
    print("✅ 组件初始化完成")
    
    # 测试数据获取（参考健康检查中的模式）
    print("获取市场数据...")
    data = market_data.get_market_data('BTC/USDT')  # 这是健康检查中使用的方法
    
    if data is not None and len(data) > 0:
        print(f"✅ 数据获取成功: {len(data)}条")
        
        # 测试信号生成
        print("生成交易信号...")
        signals = detector.analyze(data)  # 健康检查中使用的方法
        
        if signals:
            print(f"✅ 信号生成成功: {len(signals)}个信号")
        else:
            print("⚠️ 无信号生成")
            
        # 测试生产系统
        print("初始化生产系统...")
        production.initialize()
        print("✅ 生产系统就绪")
        
        return "所有组件工作正常"
    else:
        return "数据获取失败"

if __name__ == "__main__":
    result = health_check_based_test()
    print(f"\n📊 测试结果: {result}")