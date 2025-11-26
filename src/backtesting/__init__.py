# src/strategies/__init__.py
import pandas as pd
import logging
from functools import wraps

def validate_market_data(required_cols=['open', 'high', 'low', 'close']):
    """数据验证装饰器 - 简单解决 'high' 错误"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, data, *args, **kwargs):
            # 检查数据列
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                self.logger.warning(f"数据缺少列 {missing_cols}，使用默认值填充")
                
                # 自动填充缺失列
                for col in missing_cols:
                    if col == 'high':
                        data['high'] = data['close'] * 1.01  # 默认high比close高1%
                    elif col == 'low':
                        data['low'] = data['close'] * 0.99   # 默认low比close低1%
                    elif col == 'open':
                        data['open'] = data['close']         # 默认open等于close
                    elif col == 'volume':
                        data['volume'] = 10000               # 默认成交量
            
            # 检查数据长度
            if len(data) < 20:
                self.logger.warning(f"数据长度不足 ({len(data)})，可能影响策略计算")
            
            return func(self, data, *args, **kwargs)
        return wrapper
    return decorator