# web_dashboard_fixed.py
#!/usr/bin/env python3
from flask import Flask, render_template, jsonify
import threading
import time
import json

app = Flask(__name__)

class DashboardData:
    """仪表板数据"""
    def __init__(self):
        self.performance = {
            'total_return': -16.41,  # 从回测结果获取
            'today_pnl': 0.0,
            'active_strategies': ['SMA_Sensitive', 'RSI_Sensitive'],
            'current_regime': 'low_volatility',
            'system_status': '运行中',
            'total_trades': 69
        }
    
    def update(self, new_data):
        """更新数据"""
        self.performance.update(new_data)

# 全局数据
dashboard_data = DashboardData()

@app.route('/')
def index():
    """主页面"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>量化交易监控</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .card {{ border: 1px solid #ddd; padding: 20px; margin: 10px; border-radius: 8px; }}
            .positive {{ color: green; }}
            .negative {{ color: red; }}
            .status-running {{ color: green; }}
            .status-stopped {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>🎯 量化交易系统监控</h1>
        
        <div class="card">
            <h2>📊 性能概览</h2>
            <p>总收益: <span class="{'positive' if dashboard_data.performance['total_return'] > 0 else 'negative'}">
                {dashboard_data.performance['total_return']}%</span></p>
            <p>今日盈亏: {dashboard_data.performance['today_pnl']}</p>
            <p>总交易数: {dashboard_data.performance['total_trades']}</p>
        </div>
        
        <div class="card">
            <h2>🔧 系统状态</h2>
            <p>市场状态: {dashboard_data.performance['current_regime']}</p>
            <p>系统状态: <span class="status-running">{dashboard_data.performance['system_status']}</span></p>
            <p>活跃策略: {', '.join(dashboard_data.performance['active_strategies'])}</p>
        </div>
        
        <div class="card">
            <h2>📈 实时数据</h2>
            <p>数据更新: <span id="updateTime">刚刚</span></p>
            <button onclick="location.reload()">🔄 刷新</button>
        </div>
    </body>
    </html>
    """

@app.route('/api/performance')
def api_performance():
    """性能数据API"""
    return jsonify(dashboard_data.performance)

@app.route('/api/health')
def api_health():
    """健康检查API"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'components': {
            'data_pipeline': 'ok',
            'strategy_engine': 'ok', 
            'risk_management': 'ok',
            'trading_execution': 'degraded'
        }
    })

def run_dashboard():
    """运行仪表板"""
    print("🌐 启动Web监控界面: http://localhost:5000")
    print("💡 请在浏览器中访问以上地址")
    app.run(debug=False, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    # 直接运行，不使用线程
    print("🚀 启动量化交易监控面板...")
    run_dashboard()