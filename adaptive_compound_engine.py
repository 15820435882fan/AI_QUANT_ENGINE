# adaptive_compound_engine.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import logging

# 导入我们创建的策略
from src.strategies.market_regime_detector import MarketRegimeDetector, MarketRegime
from src.strategies.compound_strategy_base import CompoundStrategyBase

class AdaptiveCompoundEngine:
    """自适应复利引擎 - 核心调度系统"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.regime_detector = MarketRegimeDetector()
        self.strategies: Dict[str, CompoundStrategyBase] = {}
        self.strategy_weights = {}
        self.performance_history = []
        
        # 风险控制参数
        self.max_drawdown = 0.15  # 最大回撤15%
        self.max_position_size = 0.2  # 单策略最大仓位20%
        self.stop_loss = 0.05  # 单笔止损5%
        
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('CompoundEngine')
    
    def add_strategy(self, strategy: CompoundStrategyBase):
        """添加策略"""
        self.strategies[strategy.name] = strategy
        self.strategy_weights[strategy.name] = strategy.weight
        self.logger.info(f"添加策略: {strategy.name}, 初始权重: {strategy.weight}")
    
    def calculate_dynamic_weights(self, market_regime: MarketRegime) -> Dict[str, float]:
        """根据市场状态计算动态权重"""
        base_weights = self.strategy_weights.copy()
        
        # 根据策略偏好调整权重
        for name, strategy in self.strategies.items():
            preferred_regimes = strategy.get_preferred_regime()
            if market_regime in preferred_regimes:
                base_weights[name] *= 1.5  # 偏好状态权重提升50%
            else:
                base_weights[name] *= 0.7  # 非偏好状态权重降低30%
                
        # 根据近期表现调整权重
        for name, strategy in self.strategies.items():
            recent_perf = strategy.get_recent_performance()
            if recent_perf > 0:
                base_weights[name] *= (1 + min(recent_perf, 0.3))  # 正表现最多提升30%
            else:
                base_weights[name] *= (1 - min(abs(recent_perf), 0.2))  # 负表现最多降低20%
        
        # 归一化权重
        total_weight = sum(base_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v/total_weight for k, v in base_weights.items()}
        else:
            normalized_weights = {k: 1.0/len(base_weights) for k in base_weights.keys()}
            
        return normalized_weights
    
    def risk_management_check(self, signals: Dict[str, Any], current_portfolio_value: float) -> bool:
        """风险管理检查"""
        # 检查最大回撤
        if self.performance_history:
            peak = max(self.performance_history)
            current_drawdown = (peak - current_portfolio_value) / peak
            if current_drawdown > self.max_drawdown:
                self.logger.warning(f"触发最大回撤风控: {current_drawdown:.2%}")
                return False
        
        # 检查信号强度
        if signals.get('combined_confidence', 0) < 0.3:
            self.logger.info("信号置信度过低，跳过交易")
            return False
            
        return True
    
    def generate_compound_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成复合信号"""
        try:
            # 1. 检测市场状态
            regime_info = self.regime_detector.detect_regime(data)
            current_regime = regime_info['regime']
            
            self.logger.info(f"检测到市场状态: {current_regime.value}, 置信度: {regime_info['confidence']:.2f}")
            
            # 2. 计算动态权重
            dynamic_weights = self.calculate_dynamic_weights(current_regime)
            
            # 3. 收集各策略信号
            strategy_signals = {}
            combined_signal = 0
            confidence_sum = 0
            
            for name, strategy in self.strategies.items():
                try:
                    signals_df = strategy.calculate_signals(data)
                    if not signals_df.empty and 'signal' in signals_df.columns:
                        current_signal = signals_df['signal'].iloc[-1]
                        strategy_weight = dynamic_weights.get(name, 0.1)
                        
                        # 根据权重调整信号
                        weighted_signal = current_signal * strategy_weight
                        combined_signal += weighted_signal
                        confidence_sum += strategy_weight
                        
                        strategy_signals[name] = {
                            'signal': current_signal,
                            'weighted_signal': weighted_signal,
                            'weight': strategy_weight
                        }
                        
                except Exception as e:
                    self.logger.error(f"策略 {name} 信号计算失败: {e}")
                    continue
            
            # 4. 生成最终信号
            if confidence_sum > 0:
                final_signal = combined_signal / confidence_sum
                combined_confidence = min(confidence_sum, 1.0)
            else:
                final_signal = 0
                combined_confidence = 0
            
            # 5. 生成交易决策
            decision = self._generate_trading_decision(final_signal, combined_confidence, regime_info)
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'market_regime': current_regime.value,
                'regime_confidence': regime_info['confidence'],
                'final_signal': final_signal,
                'combined_confidence': combined_confidence,
                'strategy_signals': strategy_signals,
                'dynamic_weights': dynamic_weights,
                'decision': decision
            }
            
            self.logger.info(f"复合信号生成: 信号={final_signal:.3f}, 置信度={combined_confidence:.2f}, 决策={decision['action']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"复合信号生成失败: {e}")
            return {'error': str(e), 'decision': {'action': 'HOLD', 'confidence': 0.1}}
    
    def _generate_trading_decision(self, signal: float, confidence: float, regime_info: Dict) -> Dict[str, Any]:
        """生成交易决策"""
        # 动态阈值 - 根据市场波动率调整
        volatility_factor = regime_info.get('volatility', 0.02) / 0.02  # 归一化
        buy_threshold = 0.2 * volatility_factor
        sell_threshold = -0.2 * volatility_factor
        
        if signal > buy_threshold and confidence > 0.4:
            action = 'BUY'
            position_size = min(confidence, 0.8)  # 根据置信度决定仓位
        elif signal < sell_threshold and confidence > 0.4:
            action = 'SELL' 
            position_size = min(confidence, 0.6)  # 卖空仓位更保守
        else:
            action = 'HOLD'
            position_size = 0
            
        return {
            'action': action,
            'position_size': position_size,  # 仓位比例
            'confidence': confidence,
            'signal_strength': abs(signal),
            'volatility_adjusted': volatility_factor
        }