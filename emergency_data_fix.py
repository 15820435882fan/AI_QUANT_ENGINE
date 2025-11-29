# emergency_data_fix.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

def validate_price_data(prices):
    """验证价格数据合理性"""
    price_series = pd.Series(prices)
    
    # 检查价格范围
    if price_series.max() / price_series.min() > 1000:  # 价格变化超过1000倍
        logging.error(f"价格数据异常: 最大{price_series.max():.2f} vs 最小{price_series.min():.2f}")
        return False
    
    # 检查价格连续性
    daily_returns = price_series.pct_change().dropna()
    if (daily_returns.abs() > 1.0).any():  # 单日涨跌幅超过100%
        logging.error("发现异常价格波动")
        return False
        
    return True

def generate_realistic_btc_data(days=30, initial_price=35000):
    """生成真实的BTC模拟数据"""
    dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
    
    # 更真实的随机游走
    returns = np.random.normal(0.0005, 0.02, len(dates))  # 日均波动2%
    prices = [initial_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        # 确保价格在合理范围内
        if new_price < 1000 or new_price > 100000:
            new_price = prices[-1] * (1 + np.random.normal(0, 0.01))
        prices.append(new_price)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices], 
        'close': prices,
        'volume': np.random.uniform(1000, 50000, len(prices))
    })
    
    return df