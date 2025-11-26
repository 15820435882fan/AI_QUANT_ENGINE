# src/strategies/adaptive_strategy_engine.py
class AdaptiveStrategyEngine:
    """自适应策略引擎"""
    
    async def run_adaptive_backtest(self, historical_data: pd.DataFrame):
        """运行自适应策略回测"""
        regime_changes = []
        strategy_performance = {}
        
        # 滑动窗口分析市场状态
        window_size = 100  # 100个数据点窗口
        for i in range(window_size, len(historical_data)):
            window_data = historical_data.iloc[i-window_size:i]
            
            # 检测市场状态
            regime = await self.regime_detector.detect_regime(window_data)
            regime_changes.append(regime)
            
            # 选择并执行策略
            strategies = await self.strategy_manager.select_strategies(window_data)
            for strategy_config in strategies:
                strategy_name = strategy_config['strategy']
                if strategy_name not in strategy_performance:
                    strategy_performance[strategy_name] = {
                        'returns': [], 'trades': 0, 'active_periods': 0
                    }
                
                # 执行策略（简化）
                strategy_performance[strategy_name]['active_periods'] += 1
        
        return regime_changes, strategy_performance