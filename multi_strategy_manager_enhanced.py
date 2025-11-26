# multi_strategy_manager_enhanced.py - 完整修复版
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from src.strategies.strategy_factory import strategy_factory

class MultiStrategyManagerEnhanced:
    """增强版多策略管理器 - 带数据预处理修复"""
    
    def __init__(self):
        self.strategies: Dict[str, Any] = {}
        self.strategy_performance = {}
        
    def add_strategy(self, strategy_type: str, config: dict):
        """使用工厂添加策略"""
        try:
            strategy = strategy_factory.create_strategy(strategy_type, config)
            self.strategies[strategy.name] = strategy
            print(f"✅ 添加策略: {strategy.name}")
            return strategy
        except Exception as e:
            print(f"❌ 添加策略失败: {e}")
            return None
    
    def remove_strategy(self, strategy_name: str):
        """移除策略"""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            print(f"✅ 移除策略: {strategy_name}")
    
    def _preprocess_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        预处理市场数据 - 修复 'high' 错误的核心方法
        """
        if data.empty:
            print("⚠️ 输入数据为空")
            return data
            
        # 创建数据副本避免修改原始数据
        processed_data = data.copy()
        
        # 定义必要的OHLCV列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # 检查并修复缺失的列
        missing_cols = [col for col in required_cols if col not in processed_data.columns]
        if missing_cols:
            print(f"🔧 修复缺失数据列: {missing_cols}")
            
            # 如果有close列，基于close生成其他列
            if 'close' in processed_data.columns:
                close_prices = processed_data['close']
            else:
                # 如果没有close，尝试使用第一列数值数据
                numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    close_prices = processed_data[numeric_cols[0]]
                    processed_data['close'] = close_prices
                    print(f"🔧 使用 '{numeric_cols[0]}' 作为close价格")
                else:
                    # 最后手段：生成默认价格序列
                    close_prices = pd.Series([100] * len(processed_data), index=processed_data.index)
                    processed_data['close'] = close_prices
                    print("⚠️ 无法确定价格列，使用默认价格100")
            
            # 基于close价格生成缺失的OHLC列
            for col in missing_cols:
                if col == 'open':
                    processed_data['open'] = close_prices
                elif col == 'high':
                    # high = close * (1 + 随机0-2%)
                    processed_data['high'] = close_prices * (1 + np.random.uniform(0, 0.02, len(close_prices)))
                elif col == 'low':
                    # low = close * (1 - 随机0-2%)
                    processed_data['low'] = close_prices * (1 - np.random.uniform(0, 0.02, len(close_prices)))
                elif col == 'volume':
                    # 默认成交量
                    processed_data['volume'] = np.random.randint(1000, 100000, len(close_prices))
        
        # 确保数据类型正确
        for col in required_cols:
            if col in processed_data.columns:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
        
        # 填充可能的NaN值
        processed_data = processed_data.ffill().bfill()
        
        print(f"✅ 数据预处理完成: {len(processed_data)} 行, {len(processed_data.columns)} 列")
        return processed_data
    
    def calculate_combined_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算组合信号 - 包含数据预处理"""
        if not self.strategies:
            print("⚠️ 没有可用的策略")
            return pd.DataFrame()
            
        # 关键修复：预处理数据
        processed_data = self._preprocess_market_data(data)
        
        if processed_data.empty:
            print("❌ 预处理后数据为空，无法计算信号")
            return pd.DataFrame()
        
        combined_signals = pd.DataFrame(index=processed_data.index)
        
        for name, strategy in self.strategies.items():
            try:
                print(f"🔧 计算 {name} 信号...")
                signals = strategy.calculate_signals(processed_data)
                
                if not signals.empty and 'signal' in signals.columns:
                    combined_signals[f'{name}_signal'] = signals['signal']
                    print(f"✅ {name} 信号计算完成")
                else:
                    print(f"⚠️ {name} 返回空信号或缺少signal列")
                    
            except Exception as e:
                print(f"❌ {name} 信号计算失败: {e}")
        
        # 计算综合信号
        if not combined_signals.empty:
            signal_columns = [col for col in combined_signals.columns if 'signal' in col]
            if signal_columns:
                combined_signals['combined_signal'] = combined_signals[signal_columns].mean(axis=1)
                print(f"📊 综合信号计算完成，使用策略: {len(signal_columns)}个")
            else:
                print("⚠️ 没有有效的信号列可用于计算综合信号")
                combined_signals['combined_signal'] = 0.0
        else:
            print("❌ 所有策略都未能生成信号")
            combined_signals = pd.DataFrame(index=processed_data.index)
            combined_signals['combined_signal'] = 0.0
        
        return combined_signals
    
    def get_strategies_info(self) -> Dict[str, Any]:
        """获取所有策略信息"""
        return {
            name: strategy.get_strategy_info()
            for name, strategy in self.strategies.items()
        }
    
    def get_available_strategy_types(self) -> List[str]:
        """获取可用的策略类型"""
        return strategy_factory.get_available_strategies()['all']


# 测试数据预处理
def test_data_preprocessing():
    """测试数据预处理功能"""
    print("🧪 测试数据预处理...")
    
    manager = MultiStrategyManagerEnhanced()
    
    # 创建残缺的测试数据（模拟 'high' 错误的情况）
    bad_data = pd.DataFrame({
        'close': [100, 101, 99, 102, 98],
        'volume': [1000, 2000, 1500, 3000, 1200]
    })
    # 故意缺少 'open', 'high', 'low' 列
    
    print(f"原始数据列: {bad_data.columns.tolist()}")
    
    # 测试预处理
    processed_data = manager._preprocess_market_data(bad_data)
    print(f"处理后数据列: {processed_data.columns.tolist()}")
    
    # 检查是否修复了缺失列
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in processed_data.columns]
    
    if not missing:
        print("✅ 数据预处理成功！所有必要列都已修复")
        print(f"数据样例:")
        print(processed_data.head())
    else:
        print(f"❌ 数据预处理失败，仍然缺失: {missing}")

if __name__ == "__main__":
    test_data_preprocessing()