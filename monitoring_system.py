# monitoring_system.py - 修复导入问题
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any

class TradingMonitor:
    """交易系统监控器"""
    
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.performance_metrics = {}
        self.alert_history = []
        self.setup_monitoring()
    
    def setup_monitoring(self):
        """设置监控"""
        self.monitor_logger = logging.getLogger('trading_monitor')
        self.start_time = datetime.now()
    
    def check_system_health(self) -> Dict[str, bool]:
        """检查系统健康状态"""
        health_checks = {}
        
        try:
            # 检查策略管理器
            status = self.trading_system.get_system_status()
            health_checks['strategy_manager'] = status['status'] == 'RUNNING'
            health_checks['active_strategies'] = status['active_strategies'] > 0
            
            # 检查信号生成
            test_data = self._generate_test_data()
            signals = self.trading_system.strategy_manager.calculate_combined_signals(test_data)
            health_checks['signal_generation'] = not signals.empty
            
            # 记录性能指标
            self._record_performance_metrics(health_checks)
            
            # 检查是否需要报警
            self._check_alerts(health_checks)
            
            health_checks['overall'] = all(health_checks.values())
            
        except Exception as e:
            health_checks['overall'] = False
            self._send_alert(f"系统健康检查失败: {e}")
        
        return health_checks
    
    def _check_alerts(self, health_checks: Dict[str, bool]):
        """检查并发送报警"""
        alerts = []
        
        if not health_checks.get('active_strategies', False):
            alerts.append("没有活跃策略")
        
        if not health_checks.get('signal_generation', False):
            alerts.append("信号生成失败")
        
        # 发送报警
        for alert in alerts:
            self._send_alert(alert)
            self.alert_history.append({
                'timestamp': datetime.now(),
                'alert': alert
            })
    
    def _send_alert(self, message: str):
        """发送报警（简化版）"""
        alert_msg = f"[交易系统报警] {datetime.now()}: {message}"
        print(f"ALERT: {alert_msg}")
        self.monitor_logger.warning(alert_msg)
    
    def _generate_test_data(self):
        """生成测试数据"""
        import pandas as pd
        import numpy as np
        
        return pd.DataFrame({
            'open': np.random.normal(100, 5, 50),
            'high': np.random.normal(105, 5, 50),
            'low': np.random.normal(95, 5, 50),
            'close': np.random.normal(100, 5, 50),
            'volume': np.random.randint(1000, 10000, 50)
        })
    
    def _record_performance_metrics(self, health_checks: Dict[str, bool]):
        """记录性能指标"""
        self.performance_metrics[datetime.now()] = {
            'health_checks': health_checks,
            'active_strategies': len(self.trading_system.strategy_manager.strategies),
            'uptime': (datetime.now() - self.start_time).total_seconds()
        }
    
    def get_monitoring_report(self) -> Dict[str, Any]:
        """获取监控报告"""
        health_status = self.check_system_health()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_uptime': str(datetime.now() - self.start_time),
            'health_status': health_status,
            'recent_alerts': self.alert_history[-5:],  # 最近5个报警
            'performance_metrics': {
                'total_checks': len(self.performance_metrics),
                'success_rate': sum(1 for v in self.performance_metrics.values() 
                                  if v['health_checks'].get('overall', True)) / len(self.performance_metrics)
            }
        }

def test_monitoring_system():
    """测试监控系统"""
    from production_trading_system import ProductionTradingSystem
    from test_strategies_with_real_data import generate_realistic_test_data
    
    print("测试监控系统...")
    
    # 创建交易系统（简化版本）
    trading_system = ProductionTradingSystem()
    
    # 快速初始化（不进行完整优化）
    historical_data = generate_realistic_test_data(100)
    
    # 手动添加策略（跳过优化）
    from src.strategies.strategy_factory import strategy_factory
    sma_config = {
        'name': 'SMA监控测试',
        'parameters': {'sma_fast': 10, 'sma_slow': 30}
    }
    trading_system.strategy_manager.add_strategy('SimpleMovingAverageStrategy', sma_config)
    
    # 创建监控器
    monitor = TradingMonitor(trading_system)
    
    # 运行健康检查
    health = monitor.check_system_health()
    print(f"健康检查结果: {health}")
    
    # 获取监控报告
    report = monitor.get_monitoring_report()
    print(f"监控报告: {report}")
    
    return monitor

if __name__ == "__main__":
    test_monitoring_system()