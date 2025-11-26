# main.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自主AI量化交易系统 - 项目入口
"""

import asyncio
import sys
import os

# 添加src到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.quant_engine import QuantEngine, EngineConfig

async def main():
    """主函数"""
    print("=" * 60)
    print("🚀 自主AI量化交易系统 - 启动")
    print("=" * 60)
    
    # 创建配置
    config = EngineConfig(
        exchange="binance",
        symbols=["BTC/USDT"],
        initial_balance=10000.0
    )
    
    # 创建引擎
    engine = QuantEngine(config)
    
    try:
        # 测试引擎
        await test_quant_engine()
        print("🎉 系统启动测试完成！")
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return 1
    
    return 0

async def test_quant_engine():
    """测试量化引擎"""
    config = EngineConfig()
    engine = QuantEngine(config)
    
    print("✅ 引擎创建成功")
    print(f"✅ 配置: {engine.config}")
    
    # 测试连接
    await engine._connect_exchanges()
    print(f"✅ 引擎状态: {engine.state}")

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)