# src/strategies/compound_strategy_base.py
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
from enum import Enum

class MarketRegime(Enum):
    """市场状态枚举"""
    TREND_UP = "trend_up"      # 趋势上涨
    TREND_DOWN = "trend_down"  # 趋势下跌  
    RANGING = "ranging"        # 震荡
    HIGH_VOL = "high_vol"      # 高波动
    LOW_VOL = "low_vol"        # 低波动

class CompoundStrategyBase(ABC):
    """复利策略基类 - 所有策略都继承这个"""
    
    def __init__(self, config: dict):
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.weight = config.get('weight', 1.0)
        self.performance_history = []
        
    @abstractmethod
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算信号 - 必须实现"""
        pass
    
    @abstractmethod
    def get_preferred_regime(self) -> List[MarketRegime]:
        """返回策略偏好的市场状态"""
        pass
    
    def update_performance(self, returns: float):
        """更新策略表现历史"""
        self.performance_history.append(returns)
        if len(self.performance_history) > 100:  # 保留最近100个记录
            self.performance_history.pop(0)
    
    def get_recent_performance(self) -> float:
        """获取近期表现"""
        if not self.performance_history:
            return 0.0
        return np.mean(self.performance_history[-20:])  # 最近20期平均