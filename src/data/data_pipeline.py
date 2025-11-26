# src/data/data_pipeline.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据管道 - 实时市场数据获取和分发
"""

import asyncio
import logging
import ccxt
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from enum import Enum

class DataType(Enum):
    """数据类型枚举"""
    TICKER = "ticker"      # 实时行情
    OHLCV = "ohlcv"        # K线数据
    ORDERBOOK = "orderbook" # 深度数据
    TRADES = "trades"      # 成交记录

@dataclass
class MarketData:
    """市场数据模型"""
    symbol: str
    data_type: DataType
    data: Dict[str, Any]
    timestamp: float

class DataPipeline:
    """
    数据管道 - 负责实时数据获取和分发
    基于观察者模式，向策略系统推送数据
    """
    
    def __init__(self, exchange_name: str = "binance", symbols: List[str] = None):
        self.exchange_name = exchange_name
        self.symbols = symbols or ["BTC/USDT", "ETH/USDT"]
        self.logger = logging.getLogger(__name__)
        self.exchange = None
        self.subscribers: Dict[DataType, List[Callable]] = {}
        self.is_running = False
        self._tasks: List[asyncio.Task] = []  # 任务管理
        
        # 初始化订阅者字典
        for data_type in DataType:
            self.subscribers[data_type] = []
    
    async def initialize(self):
        """初始化数据管道"""
        self.logger.info(f"🔧 初始化数据管道，交易所: {self.exchange_name}")
        
        # 创建交易所实例
        self.exchange = getattr(ccxt, self.exchange_name)({
            'enableRateLimit': True,
            'sandbox': True  # 测试环境
        })
        
        # 加载市场数据
        await self._load_markets()
        self.logger.info("✅ 数据管道初始化完成")
    
    async def _load_markets(self):
        """加载市场信息"""
        try:
            self.logger.info("📋 加载交易对信息...")
            markets = self.exchange.load_markets()
            self.logger.info(f"✅ 加载了 {len(markets)} 个交易对")
        except Exception as e:
            self.logger.error(f"❌ 加载市场信息失败: {e}")
            raise
    
    def subscribe(self, data_type: DataType, callback: Callable):
        """订阅数据更新"""
        self.subscribers[data_type].append(callback)
        self.logger.info(f"📩 新的订阅者注册: {data_type}")
    
    def unsubscribe(self, data_type: DataType, callback: Callable):
        """取消订阅"""
        if callback in self.subscribers[data_type]:
            self.subscribers[data_type].remove(callback)
            self.logger.info(f"📪 订阅者取消注册: {data_type}")
    
    async def start(self):
        """启动数据管道"""
        self.logger.info("🚀 启动数据管道...")
        self.is_running = True
        
        # 启动数据获取任务并保存任务引用
        self._tasks = [
            asyncio.create_task(self._fetch_ticker_data()),
            asyncio.create_task(self._fetch_ohlcv_data())
        ]
        
        self.logger.info("✅ 数据管道启动完成")
    
    async def stop(self):
        """停止数据管道"""
        self.logger.info("🛑 停止数据管道...")
        self.is_running = False
        
        # 取消所有任务
        for task in self._tasks:
            task.cancel()
        
        # 等待所有任务完成
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self._tasks.clear()
        self.logger.info("✅ 数据管道已完全停止")
    
    async def _fetch_ticker_data(self):
        """获取实时ticker数据"""
        self.logger.info("📈 开始获取实时行情数据...")
        
        while self.is_running:
            try:
                for symbol in self.symbols:
                    # 获取ticker数据
                    ticker = self.exchange.fetch_ticker(symbol)
                    
                    # 创建数据对象
                    market_data = MarketData(
                        symbol=symbol,
                        data_type=DataType.TICKER,
                        data=ticker,
                        timestamp=self.exchange.milliseconds()
                    )
                    
                    # 通知订阅者
                    await self._notify_subscribers(market_data)
                
                # 控制请求频率
                await asyncio.sleep(5)  # 5秒更新一次
                
            except Exception as e:
                self.logger.error(f"❌ 获取ticker数据失败: {e}")
                await asyncio.sleep(10)  # 出错后等待更久
    
    async def _fetch_ohlcv_data(self):
        """获取K线数据"""
        self.logger.info("📊 开始获取K线数据...")
        
        while self.is_running:
            try:
                for symbol in self.symbols:
                    # 获取1分钟K线
                    ohlcv = self.exchange.fetch_ohlcv(symbol, '1m', limit=1)
                    
                    if ohlcv:
                        market_data = MarketData(
                            symbol=symbol,
                            data_type=DataType.OHLCV, 
                            data=ohlcv[0],  # 最新一根K线
                            timestamp=self.exchange.milliseconds()
                        )
                        await self._notify_subscribers(market_data)
                
                # 1分钟更新一次
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"❌ 获取K线数据失败: {e}")
                await asyncio.sleep(60)
    
    async def _notify_subscribers(self, market_data: MarketData):
        """通知所有订阅者"""
        callbacks = self.subscribers[market_data.data_type]
        
        for callback in callbacks:
            try:
                # 如果回调是异步函数
                if asyncio.iscoroutinefunction(callback):
                    await callback(market_data)
                else:
                    callback(market_data)
            except Exception as e:
                self.logger.error(f"❌ 通知订阅者失败: {e}")

# 测试数据管道
async def test_data_pipeline():
    """测试数据管道"""
    print("🧪 测试数据管道...")
    
    # 创建数据管道
    pipeline = DataPipeline(symbols=["BTC/USDT"])
    
    # 定义数据处理器
    def handle_ticker_data(data: MarketData):
        print(f"📈 收到行情数据: {data.symbol} - 价格: {data.data['last']}")
    
    def handle_ohlcv_data(data: MarketData):
        print(f"📊 收到K线数据: {data.symbol} - 收盘价: {data.data[4]}")
    
    # 订阅数据
    pipeline.subscribe(DataType.TICKER, handle_ticker_data)
    pipeline.subscribe(DataType.OHLCV, handle_ohlcv_data)
    
    # 初始化并启动
    await pipeline.initialize()
    await pipeline.start()
    
    # 运行一段时间后停止
    print("⏳ 数据管道运行中...")
    await asyncio.sleep(10)  # 运行10秒
    
    await pipeline.stop()
    print("✅ 数据管道测试完成")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_data_pipeline())