# test_data_preprocess.py
import pandas as pd
import numpy as np
from multi_strategy_manager_enhanced import MultiStrategyManagerEnhanced

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