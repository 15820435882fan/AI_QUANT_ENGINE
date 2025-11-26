#!/bin/bash
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
