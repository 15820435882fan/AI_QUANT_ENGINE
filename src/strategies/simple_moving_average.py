# src/strategies/simple_moving_average.py - 彻底修复
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .strategy_orchestrator import BaseStrategy

class SimpleMovingAverageStrategy(BaseStrategy):
    """
    SMA策略 - 统一构造函数版本（修复版）
    """
    
    def __init__(self, config: dict, data_provider=None):
        """
        修复：确保正确调用父类构造函数
        """
        print(f"🔧 SMA策略初始化开始，config: {config}")
        
        # 关键修复：正确调用父类构造函数
        # BaseStrategy.__init__(self, config, data_provider)  # 方式1
        super().__init__(config, data_provider)  # 方式2
        
        print("✅ 父类构造函数调用成功")
        
        # 从parameters中获取参数
        self.sma_fast = self.parameters.get('sma_fast', 20)
        self.sma_slow = self.parameters.get('sma_slow', 50)
        
        # 向后兼容
        self.fast_period = self.sma_fast
        self.slow_period = self.sma_slow
        
        print(f"✅ SMA策略初始化完成: fast={self.sma_fast}, slow={self.sma_slow}")

    def _handle_backward_compatibility(self):
        """处理SMA策略特定的向后兼容性"""
        print("🔧 处理向后兼容性...")
        
        # 如果config中直接提供了fast_period/slow_period，映射到parameters
        if 'fast_period' in self.config and 'sma_fast' not in self.parameters:
            self.parameters['sma_fast'] = self.config['fast_period']
            print("✅ 映射fast_period到sma_fast")
            
        if 'slow_period' in self.config and 'sma_slow' not in self.parameters:
            self.parameters['sma_slow'] = self.config['slow_period']
            print("✅ 映射slow_period到sma_slow")
            
        # 确保基础参数存在
        self.sma_fast = self.parameters.get('sma_fast', 20)
        self.sma_slow = self.parameters.get('sma_slow', 50)
        
        print(f"🔧 兼容性处理完成: fast={self.sma_fast}, slow={self.sma_slow}")

    def _initialize(self):
        """SMA特定初始化"""
        print("🔧 SMA特定初始化...")
        # 这里可以添加SMA特定的初始化逻辑
        pass

    @staticmethod
    def get_required_parameters() -> List[str]:
        """返回此策略需要的参数列表"""
        return ['sma_fast', 'sma_slow']
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算交易信号"""
        print("🔧 计算SMA信号...")
        
        if len(data) < self.sma_slow:
            self.logger.warning("数据长度不足，无法计算SMA")
            return pd.DataFrame()
            
        if 'close' not in data.columns:
            self.logger.error("数据缺少 'close' 列")
            return pd.DataFrame()
            
        data = data.copy()
        data['sma_fast'] = data['close'].rolling(window=self.sma_fast).mean()
        data['sma_slow'] = data['close'].rolling(window=self.sma_slow).mean()
        
        # 生成信号
        data['signal'] = 0
        data['position'] = 0
        
        data.loc[data['sma_fast'] > data['sma_slow'], 'signal'] = 1
        data.loc[data['sma_fast'] < data['sma_slow'], 'signal'] = -1
        
        data['position'] = data['signal'].diff().fillna(0)
        
        print(f"✅ SMA信号计算完成，数据形状: {data.shape}")
        return data

    def get_strategy_info(self) -> Dict[str, Any]:
        """返回策略详细信息"""
        base_info = super().get_strategy_info()
        base_info.update({
            'type': 'SMA',
            'parameters_detail': {
                'sma_fast': self.sma_fast,
                'sma_slow': self.sma_slow
            }
        })
        return base_info