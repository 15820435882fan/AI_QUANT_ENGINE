# run_diverse_market_test.py
async def test_diverse_markets():
    """测试不同市场状态下的表现"""
    print("🧪 测试多样化市场...")
    
    # 创建不同市场状态的数据
    market_scenarios = [
        ("trending_bull", generate_trending_data(0.002)),      # 上涨趋势
        ("trending_bear", generate_trending_data(-0.0015)),    # 下跌趋势  
        ("high_volatility", generate_volatile_data(0.005)),    # 高波动
        ("ranging", generate_ranging_data(0.0005)),           # 震荡
        ("low_volatility", generate_ranging_data(0.0001))     # 低波动
    ]
    
    for scenario_name, data in market_scenarios:
        print(f"\n📈 测试场景: {scenario_name}")
        
        # 运行自适应回测
        config = BacktestConfig(initial_capital=10000.0)
        adaptive_engine = AdaptiveBacktestEngine(config)
        result = await adaptive_engine.run_adaptive_backtest(data)
        
        print(f"  收益: {result['total_return']:.2%}")
        print(f"  交易数: {result['total_trades']}")
        print(f"  市场状态: {result['regime_changes']}次变化")