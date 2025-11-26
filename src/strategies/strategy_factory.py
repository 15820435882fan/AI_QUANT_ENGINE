# src/strategies/strategy_factory.py - 修复配置验证
import importlib
import inspect
from pathlib import Path
from typing import Dict, Type, Any, List
import pandas as pd
from .strategy_orchestrator import BaseStrategy

class LegacyStrategyAdapter(BaseStrategy):
    """旧策略适配器，将非BaseStrategy的策略包装成统一接口"""
    
    def __init__(self, legacy_strategy, config: dict):
        self.legacy_strategy = legacy_strategy
        # 调用父类构造函数
        super().__init__(config)
        
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """委托给旧策略计算信号"""
        return self.legacy_strategy.calculate_signals(data)
    
    def get_required_parameters(self) -> List[str]:
        """返回旧策略需要的参数"""
        return getattr(self.legacy_strategy, 'get_required_parameters', lambda: [])()

class StrategyFactory:
    """策略工厂，负责策略的动态创建和管理"""
    
    def __init__(self, strategies_dir: str = "src/strategies"):
        self.strategies_dir = Path(strategies_dir)
        self._strategy_classes: Dict[str, Type[BaseStrategy]] = {}
        self._legacy_strategy_classes: Dict[str, Any] = {}
        self._discover_strategies()
    
    def _discover_strategies(self):
        """自动发现所有可用的策略类"""
        try:
            # 手动注册已知策略类
            strategy_mapping = {
                'SimpleMovingAverageStrategy': 'simple_moving_average',
                'MACDStrategySmart': 'macd_strategy_smart', 
                'BollingerBandsStrategy': 'bollinger_bands_strategy',
                'TurtleTradingStrategy': 'turtle_trading_strategy'
            }
            
            for class_name, module_name in strategy_mapping.items():
                try:
                    module = importlib.import_module(f'src.strategies.{module_name}')
                    strategy_class = getattr(module, class_name)
                    
                    if issubclass(strategy_class, BaseStrategy):
                        self._strategy_classes[class_name] = strategy_class
                        print(f"✅ 注册策略: {class_name}")
                    else:
                        # 非BaseStrategy的策略使用适配器
                        self._legacy_strategy_classes[class_name] = strategy_class
                        print(f"🔄 注册旧策略(需适配): {class_name}")
                        
                except Exception as e:
                    print(f"❌ 加载策略 {class_name} 失败: {e}")
                    
        except Exception as e:
            print(f"❌ 策略发现过程出错: {e}")
    
    def create_strategy(self, strategy_type: str, config: dict, **kwargs):
        """
        创建策略实例
        """
        # 检查新式策略
        if strategy_type in self._strategy_classes:
            strategy_class = self._strategy_classes[strategy_type]
            
            try:
                # 修复：确保config包含所有必要字段
                validated_config = self._validate_and_fix_config(config)
                    
                instance = strategy_class(config=validated_config, **kwargs)
                
                # 验证参数
                if not instance.validate_parameters():
                    missing = [p for p in instance.get_required_parameters() 
                              if p not in instance.parameters]
                    print(f"⚠️  策略 {strategy_type} 缺少参数: {missing}")
                    
                return instance
                
            except Exception as e:
                raise RuntimeError(f"创建策略 {strategy_type} 失败: {e}")
        
        # 检查旧式策略（使用适配器）
        elif strategy_type in self._legacy_strategy_classes:
            print(f"🔄 使用适配器创建旧策略: {strategy_type}")
            return self._create_legacy_strategy(strategy_type, config, **kwargs)
        else:
            available = list(self._strategy_classes.keys()) + list(self._legacy_strategy_classes.keys())
            raise ValueError(f"未知策略类型: {strategy_type}。可用策略: {available}")
    
    def _validate_and_fix_config(self, config: dict) -> dict:
        """验证和修复配置字典"""
        validated = config.copy()
        
        # 确保必要字段存在
        if 'name' not in validated:
            validated['name'] = 'Unnamed_Strategy'
        if 'parameters' not in validated:
            validated['parameters'] = {}
        if 'symbols' not in validated:
            validated['symbols'] = ['BTC/USDT']
            
        return validated
    
    def _create_legacy_strategy(self, strategy_type: str, config: dict, **kwargs):
        """创建旧策略实例（使用适配器包装）"""
        legacy_class = self._legacy_strategy_classes[strategy_type]
        
        try:
            # 旧策略的创建方式（直接实例化）
            legacy_config = {
                'name': config.get('name', strategy_type),
                'symbols': config.get('symbols', ['BTC/USDT'])
            }
            
            legacy_instance = legacy_class(**legacy_config)
            
            # 使用适配器包装
            adapter = LegacyStrategyAdapter(legacy_instance, config)
            print(f"✅ 旧策略适配成功: {strategy_type}")
            return adapter
            
        except Exception as e:
            raise RuntimeError(f"创建旧策略 {strategy_type} 失败: {e}")
    
    def get_available_strategies(self):
        """获取所有可用的策略类型"""
        new_strategies = list(self._strategy_classes.keys())
        legacy_strategies = list(self._legacy_strategy_classes.keys())
        return {
            'new_strategies': new_strategies,
            'legacy_strategies': legacy_strategies,
            'all': new_strategies + legacy_strategies
        }

# 全局工厂实例
strategy_factory = StrategyFactory()