# production_trading_system.py - 修复日志编码
import pandas as pd
import numpy as np
import logging
import sys
from datetime import datetime
from typing import Dict, List, Any
from multi_strategy_manager_enhanced import MultiStrategyManagerEnhanced
from ai_strategy_optimizer_enhanced import EnhancedAIStrategyOptimizer

class ProductionTradingSystem:
    """生产环境交易系统"""
    
    def __init__(self):
        self.strategy_manager = MultiStrategyManagerEnhanced()
        self.optimizer = EnhancedAIStrategyOptimizer()
        self.optimized_strategies = {}
        self.setup_logging()
        
    def setup_logging(self):
        """设置生产环境日志（修复编码问题）"""
        # 移除emoji字符或使用兼容编码
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # 创建文件处理器（使用UTF-8编码）
        file_handler = logging.FileHandler('trading_system.log', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器（简化输出避免编码问题）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 创建格式化器（简化格式）
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 清除现有处理器并添加新处理器
        logger.handlers.clear()
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        self.logger = logging.getLogger(__name__)
    
    def initialize_optimized_strategies(self, historical_data: pd.DataFrame):
        """初始化优化后的策略"""
        self.logger.info("初始化优化策略...")
        
        # 基于优化结果的最佳组合
        best_combination = ['SimpleMovingAverageStrategy', 'MACDStrategySmart']
        
        # 优化每个策略的参数
        for strategy_type in best_combination:
            self.logger.info(f"优化 {strategy_type}...")
            best_params, best_score = self.optimizer.optimize_strategy_parameters(
                strategy_type, historical_data, generations=10, population_size=10  # 减少代数加快测试
            )
            
            # 使用优化参数创建策略
            config = {
                'name': f'优化_{strategy_type}',
                'parameters': best_params
            }
            strategy = self.strategy_manager.add_strategy(strategy_type, config)
            self.optimized_strategies[strategy_type] = {
                'strategy': strategy,
                'parameters': best_params,
                'score': best_score
            }
            
            self.logger.info(f"{strategy_type} 优化完成: 得分={best_score:.4f}")
    
    def process_market_data(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """处理市场数据并生成交易信号"""
        self.logger.info(f"处理市场数据，形状: {market_data.shape}")
        
        try:
            # 计算组合信号
            combined_signals = self.strategy_manager.calculate_combined_signals(market_data)
            
            if combined_signals.empty:
                return {'error': '无有效信号'}
            
            # 生成交易决策
            latest_signal = combined_signals['combined_signal'].iloc[-1]
            decision = self._make_trading_decision(latest_signal, combined_signals)
            
            self.logger.info(f"交易决策: {decision}")
            return decision
            
        except Exception as e:
            self.logger.error(f"信号处理失败: {e}")
            return {'error': str(e)}
    
    def _make_trading_decision(self, signal: float, signals_df: pd.DataFrame) -> Dict[str, Any]:
        """基于信号生成交易决策"""
        # 信号强度分析
        signal_strength = abs(signal)
        signal_trend = signals_df['combined_signal'].tail(5).mean()
        
        # 交易决策逻辑
        if signal > 0.3:
            action = 'BUY'
            confidence = min(signal_strength * 2, 1.0)
        elif signal < -0.3:
            action = 'SELL' 
            confidence = min(signal_strength * 2, 1.0)
        else:
            action = 'HOLD'
            confidence = 0.1
        
        return {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'confidence': round(confidence, 3),
            'signal_strength': round(signal_strength, 3),
            'signal_trend': round(signal_trend, 3),
            'strategies_used': len(self.strategy_manager.strategies)
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        strategies_info = self.strategy_manager.get_strategies_info()
        
        return {
            'status': 'RUNNING',
            'active_strategies': len(self.strategy_manager.strategies),
            'optimized_strategies': list(self.optimized_strategies.keys()),
            'last_optimization': datetime.now().isoformat(),
            'strategies_detail': strategies_info
        }

def test_production_system():
    """测试生产交易系统"""
    print("测试生产交易系统...")
    
    # 创建系统实例
    trading_system = ProductionTradingSystem()
    
    # 生成历史数据用于优化
    from test_strategies_with_real_data import generate_realistic_test_data
    historical_data = generate_realistic_test_data(200)  # 减少数据量加快测试
    
    # 初始化优化策略
    trading_system.initialize_optimized_strategies(historical_data)
    
    # 测试实时数据处理
    realtime_data = generate_realistic_test_data(50)
    decision = trading_system.process_market_data(realtime_data)
    
    print(f"交易决策: {decision}")
    
    # 检查系统状态
    status = trading_system.get_system_status()
    print(f"系统状态: {status}")
    
    return trading_system

if __name__ == "__main__":
    test_production_system()