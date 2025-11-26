# deploy_production.py - 简化版本（使用JSON）
import os
import json
from datetime import datetime

def create_production_config():
    """创建生产环境配置文件（使用JSON）"""
    
    config = {
        'trading_system': {
            'name': 'AI量化交易系统生产版',
            'version': '1.0.0',
            'deployment_date': datetime.now().isoformat()
        },
        'strategies': {
            'default_combination': ['SimpleMovingAverageStrategy', 'MACDStrategySmart'],
            'optimization_settings': {
                'generations': 20,
                'population_size': 15,
                'evaluation_period': 300
            }
        },
        'risk_management': {
            'max_position_size': 0.1,  # 10%
            'daily_loss_limit': 0.05,  # 5%
            'max_drawdown': 0.15       # 15%
        },
        'monitoring': {
            'health_check_interval': 60,  # 秒
            'performance_report_interval': 3600  # 小时
        },
        'logging': {
            'level': 'INFO',
            'file': 'logs/trading_system.log',
            'max_size_mb': 100
        }
    }
    
    # 保存JSON配置
    os.makedirs('config', exist_ok=True)
    with open('config/production_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("生产配置文件创建完成")
    return config

def create_deployment_script():
    """创建部署脚本"""
    
    script_content = '''#!/bin/bash
# AI量化交易系统部署脚本

echo "开始部署AI量化交易系统..."

# 创建目录结构
mkdir -p logs
mkdir -p data
mkdir -p config

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "Python未安装"
    exit 1
fi

# 检查依赖
echo "检查Python依赖..."
python -c "import pandas, numpy, logging" || {
    echo "缺少必要依赖"
    exit 1
}

# 启动交易系统
echo "启动交易系统..."
python production_trading_system.py &

echo "部署完成！交易系统正在运行..."
echo "查看日志: tail -f logs/trading_system.log"
'''

    with open('deploy.sh', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod('deploy.sh', 0o755)
    print("部署脚本创建完成")

def create_project_structure():
    """创建项目结构"""
    directories = [
        'config',
        'logs', 
        'data/historical',
        'data/realtime',
        'backups',
        'reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")

if __name__ == "__main__":
    print("准备生产环境部署...")
    
    create_project_structure()
    create_production_config()
    create_deployment_script()
    
    print("\n生产环境部署准备完成！")
    print("下一步:")
    print("1. 运行: chmod +x deploy.sh")
    print("2. 执行: ./deploy.sh")
    print("3. 监控: tail -f logs/trading_system.log")