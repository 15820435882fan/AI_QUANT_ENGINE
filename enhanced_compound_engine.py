# enhanced_compound_engine.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import logging

# 正确导入基础引擎
from adaptive_compound_engine import AdaptiveCompoundEngine
from src.strategies.market_regime_detector import MarketRegimeDetector, MarketRegime

class RiskEnhancedEngine:
    """增强风险控制引擎"""
    
    def __init__(self):
        self.stop_loss_pct = 0.08  # 8%止损
        self.take_profit_pct = 0.15  # 15%止盈
        self.trailing_stop_pct = 0.05  # 5%移动止损
        self.max_position_hold_days = 30  # 最大持仓30天
        
    def check_stop_loss(self, current_price: float, entry_price: float, 
                       position_type: str) -> Tuple[bool, str]:
        """检查止损条件"""
        if position_type == 'LONG':
            pnl_pct = (current_price - entry_price) / entry_price
            if pnl_pct <= -self.stop_loss_pct:
                return True, f"止损触发: 亏损{pnl_pct:.2%}"
            elif pnl_pct >= self.take_profit_pct:
                return True, f"止盈触发: 盈利{pnl_pct:.2%}"
        else:  # SHORT
            pnl_pct = (entry_price - current_price) / entry_price
            if pnl_pct <= -self.stop_loss_pct:
                return True, f"止损触发: 亏损{pnl_pct:.2%}"
            elif pnl_pct >= self.take_profit_pct:
                return True, f"止盈触发: 盈利{pnl_pct:.2%}"
                
        return False, ""

class DynamicThresholdManager:
    """动态阈值管理器"""
    
    def __init__(self):
        self.market_volatility = 0.02
        self.base_buy_threshold = 0.1
        self.base_sell_threshold = -0.1
        
    def calculate_dynamic_thresholds(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算动态交易阈值"""
        if data.empty or len(data) < 20:
            return {
                'buy_threshold': 0.1,
                'sell_threshold': -0.1,
                'volatility_factor': 1.0
            }
            
        closes = data['close'].values
        
        # 计算当前波动率
        returns = np.diff(closes) / closes[:-1]
        current_volatility = np.std(returns) if len(returns) > 10 else 0.02
        
        # 波动率调整因子
        vol_factor = current_volatility / self.market_volatility
        
        # 动态调整阈值
        buy_threshold = self.base_buy_threshold * vol_factor
        sell_threshold = self.base_sell_threshold * vol_factor
        
        # 确保阈值在合理范围内
        buy_threshold = max(0.05, min(buy_threshold, 0.3))
        sell_threshold = min(-0.05, max(sell_threshold, -0.3))
        
        return {
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'volatility_factor': vol_factor
        }

class EnhancedCompoundEngine(AdaptiveCompoundEngine):
    """增强版复利引擎"""
    
    def __init__(self, initial_capital: float = 10000.0):
        # 先初始化父类
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.regime_detector = MarketRegimeDetector()
        self.strategies: Dict[str, Any] = {}
        self.strategy_weights = {}
        self.performance_history = []
        
        # 风险控制参数
        self.max_drawdown = 0.15  # 最大回撤15%
        self.max_position_size = 0.2  # 单策略最大仓位20%
        self.stop_loss = 0.05  # 单笔止损5%
        
        self.setup_logging()
        
        # 增强组件
        self.risk_engine = RiskEnhancedEngine()
        self.threshold_manager = DynamicThresholdManager()
        self.trade_count = 0
        self.max_trades_per_month = 20  # 每月最大交易次数
        self.last_trade_month = datetime.now().month
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('EnhancedCompoundEngine')
    
    def add_strategy(self, strategy):
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
    
    def _generate_trading_decision(self, signal: float, confidence: float, regime_info: Dict) -> Dict[str, Any]:
        """增强版交易决策"""
        # 月度交易计数重置
        current_month = datetime.now().month
        if current_month != self.last_trade_month:
            self.trade_count = 0
            self.last_trade_month = current_month
        
        # 获取动态阈值
        sample_data = pd.DataFrame({
            'close': [100] * 50  # 简化数据用于计算阈值
        })
        thresholds = self.threshold_manager.calculate_dynamic_thresholds(sample_data)
        
        buy_threshold = thresholds['buy_threshold']
        sell_threshold = thresholds['sell_threshold']
        
        # 交易频率控制
        if self.trade_count >= self.max_trades_per_month:
            return {
                'action': 'HOLD',
                'position_size': 0,
                'confidence': confidence,
                'reason': '达到月度交易上限',
                'signal_strength': abs(signal),
                'volatility_adjusted': thresholds['volatility_factor']
            }
        
        # 降低阈值要求，提高交易频率
        if signal > buy_threshold and confidence > 0.25:  # 降低置信度要求
            action = 'BUY'
            position_size = min(confidence * 0.8, 0.6)  # 更保守的仓位
            self.trade_count += 1
            reason = f"信号强度{signal:.3f} > 买入阈值{buy_threshold:.3f}"
        elif signal < sell_threshold and confidence > 0.25:
            action = 'SELL'
            position_size = min(confidence * 0.6, 0.4)  # 卖空更保守
            self.trade_count += 1
            reason = f"信号强度{signal:.3f} < 卖出阈值{sell_threshold:.3f}"
        else:
            action = 'HOLD'
            position_size = 0
            reason = "信号强度不足"
            
        return {
            'action': action,
            'position_size': position_size,
            'confidence': confidence,
            'signal_strength': abs(signal),
            'volatility_adjusted': thresholds['volatility_factor'],
            'dynamic_thresholds': thresholds,
            'reason': reason,
            'monthly_trades_remaining': self.max_trades_per_month - self.trade_count
        }
    
    def generate_compound_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成复合信号 - 重写以包含增强逻辑"""
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
            
            # 5. 生成交易决策（使用增强版）
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
            
            self.logger.info(f"增强版信号生成: 信号={final_signal:.3f}, 置信度={combined_confidence:.2f}, 决策={decision['action']}")
            self.logger.info(f"月度交易剩余: {decision.get('monthly_trades_remaining', 'N/A')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"复合信号生成失败: {e}")
            return {'error': str(e), 'decision': {'action': 'HOLD', 'confidence': 0.1}}