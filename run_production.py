# run_production.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生产环境启动脚本 - 包含Web监控界面
"""

import asyncio
import logging
import sys
import os
import uvicorn

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.core.complete_engine import CompleteTradingEngine
from src.web.dashboard import app, Dashboard

async def main():
    """主函数"""
    print("=" * 60)
    print("🚀 AI量化交易系统 - 生产环境启动")
    print("=" * 60)
    
    # 创建交易引擎
    engine = CompleteTradingEngine(symbols=["BTC/USDT", "ETH/USDT"])
    
    # 创建监控面板
    dashboard = Dashboard(engine)
    
    try:
        # 启动交易引擎
        engine_task = asyncio.create_task(engine.start())
        
        # 启动Web服务器
        print("🌐 启动Web监控面板: http://localhost:8000")
        config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        
        await server.serve()
        
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