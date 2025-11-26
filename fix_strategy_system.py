# fix_strategy_system.py
#!/usr/bin/env python3
import sys
import os
import pandas as pd
from typing import Dict, Any

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def fix_strategy_system():
    """修复策略系统数据字段访问问题"""
    print("🔧 修复策略系统数据字段访问...")
    
    # 检查数据管道
    try:
        from src.data.data_pipeline import MarketData, DataType
        
        # 创建测试数据验证字段访问
        test_data = MarketData(
            symbol="BTC/USDT",
            data_type=DataType.OHLCV,
            data=[1620000000000, 50000.0, 50500.0, 49500.0, 50200.0, 1000.0],  # timestamp, open, high, low, close, volume
            timestamp=1620000000000
        )
        
        # 测试字段访问
        if hasattr(test_data, 'data') and len(test_data.data) >= 4:
            close_price = test_data.data[4]  # 收盘价在索引4
            print(f"✅ 数据字段访问正常 - 收盘价: {close_price}")
        else:
            print("❌ 数据格式异常")
            
    except Exception as e:
        print(f"❌ 策略系统修复失败: {e}")
        # 创建兼容性补丁
        create_compatibility_patch()

def create_compatibility_patch():
    """创建数据兼容性补丁"""
    patch_code = '''
# data_compatibility.py
import sys
from typing import List, Dict, Any

class DataCompatibility:
    """数据兼容性层 - 统一数据字段访问"""
    
    @staticmethod
    def get_close_price(market_data) -> float:
        """统一获取收盘价"""
        try:
            if hasattr(market_data, 'close'):
                return float(market_data.close)
            elif hasattr(market_data, 'data'):
                data = market_data.data
                if isinstance(data, (list, tuple)) and len(data) >= 5:
                    return float(data[4])  # OHLCV格式: [timestamp, open, high, low, close, volume]
                elif isinstance(data, dict) and 'close' in data:
                    return float(data['close'])
            elif hasattr(market_data, 'price'):
                return float(market_data.price)
        except (ValueError, TypeError, IndexError) as e:
            print(f"收盘价提取错误: {e}")
        return None
    
    @staticmethod
    def get_high_price(market_data) -> float:
        """统一获取最高价"""
        try:
            if hasattr(market_data, 'high'):
                return float(market_data.high)
            elif hasattr(market_data, 'data'):
                data = market_data.data
                if isinstance(data, (list, tuple)) and len(data) >= 4:
                    return float(data[2])  # OHLCV格式: high在索引2
                elif isinstance(data, dict) and 'high' in data:
                    return float(data['high'])
        except (ValueError, TypeError, IndexError) as e:
            print(f"最高价提取错误: {e}")
        return None

# 全局实例
data_comp = DataCompatibility()
'''
    
    with open('data_compatibility.py', 'w', encoding='utf-8') as f:
        f.write(patch_code)
    print("✅ 已创建数据兼容性补丁")

if __name__ == "__main__":
    fix_strategy_system()