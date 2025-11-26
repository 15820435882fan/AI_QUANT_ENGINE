# config.py
import os
from typing import Dict, Any

# 交易配置
TRADING_CONFIG = {
    'symbols': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT'],
    'max_positions': 3,
    'base_position_size': 0.05,  # 5%
    'max_position_size': 0.20,   # 20%
    'leverage_range': [5, 20],
    'stop_loss_pct': 0.02,
    'take_profit_pct': 0.06,
    'volume_threshold': 3.0,
    'price_threshold': 0.025,
    'min_confidence': 0.75
}

# 风险控制
RISK_MANAGEMENT = {
    'max_daily_loss': 0.05,      # 单日最大亏损5%
    'max_drawdown': 0.15,        # 最大回撤15%
    'risk_per_trade': 0.02,      # 单笔风险2%
    'cooldown_after_loss': 300   # 亏损后冷却5分钟
}

def setup_environment():
    """设置环境变量"""
    required_vars = ['BINANCE_API_KEY', 'BINANCE_SECRET']
    
    for var in required_vars:
        if not os.getenv(var):
            print(f"⚠️  请设置环境变量: {var}")
            print(f"   在Windows: set {var}=your_value")
            print(f"   在Linux/Mac: export {var}=your_value")
            return False
    
    return True