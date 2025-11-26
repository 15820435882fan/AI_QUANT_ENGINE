# src/strategies/strategy_orchestrator.py
from typing import Dict, Any, List, Optional
import pandas as pd
import logging

class TradingSignal:
    """交易信号类 - 修复导入错误"""
    def __init__(self, action: str, confidence: float, strength: float = 0.0):
        self.action = action  # BUY, SELL, HOLD
        self.confidence = confidence
        self.strength = strength
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action,
            'confidence': self.confidence,
            'strength': self.strength
        }
from enum import Enum
class SignalType(Enum):
    """信号类型枚举 - 修复导入错误"""
    BUY = "BUY"
    SELL = "SELL" 
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"
class BaseStrategy:
    """
    所有策略的基类，强制统一的配置接口。
    """
    def __init__(self, config: dict, data_provider=None):
        # 保持您之前的BaseStrategy代码不变
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.parameters = config.get('parameters', {})
        self.symbols = config.get('symbols', ['BTC/USDT'])
        self.data_provider = data_provider
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._handle_backward_compatibility()
        self._initialize()
        self.logger.info(f"策略 {self.name} 初始化完成")

    def _handle_backward_compatibility(self):
        """处理向后兼容性"""
        legacy_params = ['fast_period', 'slow_period', 'period', 'std_dev', 
                        'entry_period', 'exit_period', 'atr_period', 'risk_per_trade']
        for param in legacy_params:
            if hasattr(self, param) and param not in self.parameters:
                self.parameters[param] = getattr(self, param)

    def _initialize(self):
        """策略初始化钩子"""
        pass

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算交易信号 - 必须由子类实现
        """
        raise NotImplementedError("子类必须实现 calculate_signals 方法")

    def get_required_parameters(self) -> List[str]:
        """返回此策略需要的参数列表"""
        return []

    def validate_parameters(self) -> bool:
        """验证参数是否完整"""
        required_params = self.get_required_parameters()
        return all(param in self.parameters for param in required_params)

    def get_strategy_info(self) -> Dict[str, Any]:
        """返回策略信息"""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'symbols': self.symbols,
            'parameters_valid': self.validate_parameters(),
            'class_name': self.__class__.__name__
        }