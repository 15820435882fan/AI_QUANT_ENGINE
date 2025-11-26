# start_production.py
import os
import time
import logging
from production_trading_system import ProductionTradingSystem
from monitoring_system import TradingMonitor
from real_market_data import RealMarketData

class ProductionStarter:
    """生产环境启动器"""
    
    def __init__(self):
        self.setup_logging()
        self.trading_system = None
        self.monitor = None
        self.market_data = RealMarketData()
    
    def setup_logging(self):
        """设置启动日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('production_start.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def start_production_system(self):
        """启动生产系统"""
        self.logger.info("🚀 启动AI量化交易生产系统...")
        
        try:
            # 1. 初始化交易系统
            self.logger.info("步骤1: 初始化交易系统...")
            self.trading_system = ProductionTradingSystem()
            
            # 2. 获取历史数据并优化策略
            self.logger.info("步骤2: 获取市场数据并优化策略...")
            historical_data = self.market_data.get_binance_data('BTC-USDT', limit=300)
            self.trading_system.initialize_optimized_strategies(historical_data)
            
            # 3. 启动监控系统
            self.logger.info("步骤3: 启动监控系统...")
            self.monitor = TradingMonitor(self.trading_system)
            
            # 4. 系统状态检查
            self.logger.info("步骤4: 系统状态检查...")
            system_status = self.trading_system.get_system_status()
            health_status = self.monitor.check_system_health()
            
            self.logger.info(f"✅ 交易系统状态: {system_status['status']}")
            self.logger.info(f"✅ 健康检查结果: {health_status['overall']}")
            self.logger.info(f"✅ 活跃策略: {system_status['active_strategies']}个")
            
            # 5. 进入主循环
            self.logger.info("步骤5: 进入主交易循环...")
            self._main_loop()
            
        except Exception as e:
            self.logger.error(f"❌ 系统启动失败: {e}")
            raise
    
    def _main_loop(self):
        """主交易循环"""
        self.logger.info("开始主交易循环...")
        
        iteration = 0
        while True:
            try:
                iteration += 1
                self.logger.info(f"--- 交易周期 {iteration} ---")
                
                # 获取实时数据
                realtime_data = self.market_data.get_binance_data('BTC-USDT', limit=50)
                
                # 处理市场数据
                decision = self.trading_system.process_market_data(realtime_data)
                
                # 记录决策
                self.logger.info(f"交易决策: {decision}")
                
                # 健康检查
                if iteration % 10 == 0:
                    health = self.monitor.check_system_health()
                    self.logger.info(f"定期健康检查: {health}")
                
                # 等待下一个周期（模拟实时交易）
                time.sleep(60)  # 1分钟周期
                
                # 测试运行，只运行5个周期
                if iteration >= 5:
                    self.logger.info("测试运行完成，退出主循环")
                    break
                    
            except KeyboardInterrupt:
                self.logger.info("用户中断，停止系统...")
                break
            except Exception as e:
                self.logger.error(f"主循环错误: {e}")
                time.sleep(10)  # 错误后等待10秒
    
    def get_system_summary(self):
        """获取系统摘要"""
        if not self.trading_system or not self.monitor:
            return {"status": "NOT_STARTED"}
        
        system_status = self.trading_system.get_system_status()
        monitor_report = self.monitor.get_monitoring_report()
        
        return {
            "trading_system": system_status,
            "monitoring": monitor_report,
            "timestamp": time.time()
        }

def main():
    """主启动函数"""
    print("=" * 50)
    print("🤖 AI量化交易系统 - 生产环境启动")
    print("=" * 50)
    
    starter = ProductionStarter()
    
    try:
        # 启动系统
        starter.start_production_system()
        
        # 显示最终摘要
        summary = starter.get_system_summary()
        print("\n" + "=" * 50)
        print("🎉 系统启动完成!")
        print("=" * 50)
        print(f"状态: {summary['trading_system']['status']}")
        print(f"策略: {summary['trading_system']['active_strategies']}个")
        print(f"运行时间: {summary['monitoring']['system_uptime']}")
        print(f"健康状态: {summary['monitoring']['health_status']['overall']}")
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main()