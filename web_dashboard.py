# web_dashboard.py
#!/usr/bin/env python3
from flask import Flask, render_template, jsonify
import asyncio
import threading
import json

app = Flask(__name__)

class DashboardData:
    """仪表板数据"""
    def __init__(self):
        self.performance = {
            'total_return': 0.0,
            'today_pnl': 0.0,
            'active_strategies': [],
            'current_regime': 'unknown'
        }
    
    def update(self, new_data):
        """更新数据"""
        self.performance.update(new_data)

# 全局数据
dashboard_data = DashboardData()

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html', data=dashboard_data.performance)

@app.route('/api/performance')
def api_performance():
    """性能数据API"""
    return jsonify(dashboard_data.performance)

@app.route('/api/trades')
def api_trades():
    """交易数据API"""
    return jsonify([])  # 返回交易记录

def run_dashboard():
    """运行仪表板"""
    print("🌐 启动Web监控界面: http://localhost:5000")
    app.run(debug=True, use_reloader=False)

if __name__ == "__main__":
    # 在后台线程中启动Web界面
    web_thread = threading.Thread(target=run_dashboard)
    web_thread.daemon = True
    web_thread.start()
    
    # 保持主程序运行
    try:
        while True:
            asyncio.sleep(1)
    except KeyboardInterrupt:
        print("🛑 停止监控系统")