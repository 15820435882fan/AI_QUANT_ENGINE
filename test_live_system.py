# test_live_system.py
#!/usr/bin/env python3
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_live_system():
    """测试实盘交易系统"""
    print("🧪 测试实盘交易系统...")
    
    try:
        from src.trading.live_trader import LiveTrader
        
        trader = LiveTrader(paper_trading=True)
        
        # 测试短时间运行
        print("🚀 启动交易引擎（运行30秒）...")
        await asyncio.wait_for(trader.start_trading(), timeout=30)
        
    except asyncio.TimeoutError:
        print("✅ 实盘系统正常启动")
    except Exception as e:
        print(f"❌ 实盘系统错误: {e}")

if __name__ == "__main__":
    asyncio.run(test_live_system())