#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境验证脚本 - 确保所有依赖正常工作
"""
import sys
import asyncio
import ccxt
from datetime import datetime

def check_environment():
    """检查基础环境"""
    print("🔍 检查AI量化交易引擎环境...")
    
    # 检查Python版本
    print(f"✅ Python版本: {sys.version}")
    
    # 检查关键依赖
    try:
        import pandas as pd
        print(f"✅ Pandas版本: {pd.__version__}")
        
        import ccxt
        print(f"✅ CCXT版本: {ccxt.__version__}")
        
        import fastapi
        print(f"✅ FastAPI版本: {fastapi.__version__}")
        
        return True
    except ImportError as e:
        print(f"❌ 依赖缺失: {e}")
        return False

async def test_exchange_connection():
    """测试交易所连接"""
    print("\n🔗 测试交易所连接...")
    try:
        exchange = ccxt.binance()
        markets = exchange.load_markets()
        print(f"✅ 币安连接成功，支持 {len(markets)} 个交易对")
        
        # 获取BTC/USDT行情
        ticker = exchange.fetch_ticker('BTC/USDT')
        print(f"✅ BTC/USDT 当前价格: {ticker['last']} USDT")
        
        return True
    except Exception as e:
        print(f"❌ 交易所连接失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 AI量化交易引擎 - 环境验证")
    print("=" * 50)
    
    # 同步检查
    env_ok = check_environment()
    
    # 异步检查
    if env_ok:
        exchange_ok = asyncio.run(test_exchange_connection())
    
    print("\n" + "=" * 50)
    if env_ok and exchange_ok:
        print("🎉 环境验证通过！可以开始开发。")
    else:
        print("❌ 环境验证失败，请检查依赖安装。")