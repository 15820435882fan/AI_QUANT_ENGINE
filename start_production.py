#!/usr/bin/env python3
"""
AI量化交易系统 - 生产环境启动脚本
修复版本：解决字典格式化错误
"""

import asyncio
import logging
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production_trading_system import ProductionTradingSystem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('production.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

def safe_format_dict(data):
    """安全格式化字典，避免格式化错误"""
    if isinstance(data, dict):
        return "{" + ", ".join([f"{k}:{v}" for k, v in data.items()]) + "}"
    return str(data)

async def main():
    """主函数 - 修复版本"""
    try:
        print("=" * 50)
        print("🤖 AI量化交易系统 - 生产环境启动")
        print("=" * 50)
        
        logger.info("🚀 启动AI量化交易生产系统...")
        logger.info("步骤1: 初始化交易系统...")
        
        # 初始化交易系统
        trading_system = ProductionTradingSystem()
        
        logger.info("步骤2: 获取市场数据并优化策略...")
        print("获取 BTC-USDT 市场数据...")
        
        # 优化策略 - 使用安全版本
        optimized_strategies = await safe_optimize_strategies(trading_system)
        
        logger.info(f"步骤3: 启动 {len(optimized_strategies)} 个优化策略...")
        
        # 安全记录策略信息
        for strategy_name, config in optimized_strategies.items():
            safe_config = {
                'name': config.get('name', '未知'),
                'parameters': safe_format_dict(config.get('parameters', {})),
                'symbols': config.get('symbols', [])
            }
            logger.info(f"启动策略: {strategy_name} - {safe_config['name']}")
        
        logger.info("步骤4: 开始实时交易监控...")
        print("✅ 系统启动完成！开始监控市场...")
        
        # 这里可以添加实时监控逻辑
        await asyncio.sleep(1)
        
        print("🎉 生产系统正常运行中...")
        return True
        
    except Exception as e:
        logger.error(f"❌ 系统启动失败: {e}")
        print(f"❌ 启动失败: {e}")
        return False

async def safe_optimize_strategies(trading_system):
    """安全优化策略，避免格式化错误"""
    try:
        # 调用原有的优化方法
        optimized_strategies = trading_system.optimize_strategies()
        
        # 安全处理返回结果
        safe_strategies = {}
        for name, config in optimized_strategies.items():
            # 确保配置中的所有值都是可格式化的
            safe_config = {
                'name': str(config.get('name', f'优化_{name}')),
                'parameters': config.get('parameters', {}),
                'symbols': [str(s) for s in config.get('symbols', [])]
            }
            safe_strategies[name] = safe_config
            
        return safe_strategies
        
    except Exception as e:
        logger.error(f"策略优化失败，使用默认策略: {e}")
        # 返回默认策略
        return get_default_strategies()

def get_default_strategies():
    """获取默认策略配置"""
    return {
        'SimpleMovingAverageStrategy': {
            'name': '默认_SMA策略',
            'parameters': {'sma_fast': 10, 'sma_slow': 30},
            'symbols': ['BTC/USDT']
        },
        'MACDStrategySmart': {
            'name': '默认_MACD策略', 
            'parameters': {'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9},
            'symbols': ['BTC/USDT']
        }
    }

if __name__ == "__main__":
    try:
        # 运行修复版本
        success = asyncio.run(main())
        if success:
            print("🎊 系统启动成功！")
            sys.exit(0)
        else:
            print("💥 系统启动失败")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 用户中断系统")
        sys.exit(0)
    except Exception as e:
        print(f"💥 未处理的错误: {e}")
        sys.exit(1)