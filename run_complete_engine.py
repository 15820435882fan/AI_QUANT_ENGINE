# run_complete_engine.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目入口 - 运行完整交易引擎
"""

import asyncio
import logging
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.core.complete_engine import CompleteTradingEngine

async def main():
    """主函数"""
    print("=" * 60)
    print("🚀 AI量化交易系统 - 完整引擎启动")
    print("=" * 60)
    
    # 创建并启动引擎
    engine = CompleteTradingEngine(symbols=["BTC/USDT"])
    
    try:
        await engine.start()
        
        # 让引擎运行一段时间
        print("⏳ 系统运行中，按 Ctrl+C 停止...")
        await asyncio.sleep(60)  # 运行60秒
        
    except KeyboardInterrupt:
        print("\n🛑 用户请求停止...")
    finally:
        await engine.stop()
    
    # 输出最终报告
    report = engine.get_status_report()
    print("\n📊 最终系统报告:")
    for key, value in report.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())