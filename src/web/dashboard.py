# src/web/dashboard.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web监控面板 - 实时监控交易状态
"""

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import asyncio
import json

app = FastAPI(title="AI量化交易监控面板")

class Dashboard:
    """
    Web监控面板
    """
    
    def __init__(self, trading_engine):
        self.engine = trading_engine
        self.connected_clients = []
        
    async def broadcast_update(self):
        """广播状态更新给所有连接的客户端"""
        status = self.engine.get_status_report()
        
        for client in self.connected_clients:
            try:
                await client.send_text(json.dumps(status))
            except:
                self.connected_clients.remove(client)
    
    @app.get("/")
    async def get_dashboard():
        """返回监控面板HTML"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI量化交易监控</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .status-card { border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 5px; }
                .buy { color: green; }
                .sell { color: red; }
                .hold { color: orange; }
            </style>
        </head>
        <body>
            <h1>🚀 AI量化交易系统监控面板</h1>
            
            <div class="status-card">
                <h3>系统状态</h3>
                <div id="system-status">加载中...</div>
            </div>
            
            <div class="status-card">
                <h3>交易信号</h3>
                <div id="signals">加载中...</div>
            </div>
            
            <div class="status-card">
                <h3>订单状态</h3>
                <div id="orders">加载中...</div>
            </div>
            
            <script>
                const ws = new WebSocket('ws://localhost:8000/ws');
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    // 更新系统状态
                    document.getElementById('system-status').innerHTML = `
                        <p>运行状态: ${data.running ? '运行中' : '已停止'}</p>
                        <p>接收数据: ${data.data_received} 条</p>
                        <p>生成信号: ${data.signals_generated} 个</p>
                        <p>执行订单: ${data.orders_executed} 个</p>
                        <p>当前状态: ${data.current_state}</p>
                    `;
                };
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        dashboard.connected_clients.append(websocket)
        
        try:
            while True:
                # 保持连接
                await asyncio.sleep(1)
        except:
            dashboard.connected_clients.remove(websocket)