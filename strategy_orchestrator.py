# strategy_orchestrator.py
from typing import Dict, Any, List, Optional
import pandas as pd
import logging

class BaseStrategy:
    """
    所有策略的基类，强制统一的配置接口。
    """
    def __init__(self, config: dict, data_provider=None):
        """
        统一的策略构造函数。
        
        Args:
            config (dict): 策略配置字典，必须包含 'name' 和 'parameters' 键。
                示例: {
                    'name': 'MyStrategy',
                    'parameters': {'sma_fast': 20, 'sma_slow': 50},
                    'symbols': ['BTC/USDT']
                }
            data_provider: 数据提供器实例
        """
        # 参数验证和设置
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary.")
        
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.parameters = config.get('parameters', {})
        self.symbols = config.get('symbols', ['BTC/USDT'])
        self.data_provider = data_provider
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 处理向后兼容性
        self._handle_backward_compatibility()
        
        # 调用初始化钩子
        self._initialize()
        
        self.logger.info(f"策略 {self.name} 初始化完成")

    def _handle_backward_compatibility(self):
        """处理向后兼容性 - 子类可重写"""
        # 基础兼容性处理：将旧式参数映射到新格式
        legacy_params = ['fast_period', 'slow_period', 'period', 'std_dev', 
                        'entry_period', 'exit_period', 'atr_period', 'risk_per_trade']
        
        for param in legacy_params:
            if hasattr(self, param) and param not in self.parameters:
                self.parameters[param] = getattr(self, param)

    def _initialize(self):
        """策略初始化钩子 - 子类可重写"""
        pass

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算交易信号 - 必须由子类实现
        """
        raise NotImplementedError("子类必须实现 calculate_signals 方法")

    def get_required_parameters(self) -> List[str]:
        """
        返回此策略需要的参数列表 - 子类应该重写
        """
        return []

    def validate_parameters(self) -> bool:
        """
        验证参数是否完整
        """
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