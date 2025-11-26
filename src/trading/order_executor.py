# src/trading/order_executor.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
订单执行器 - 负责实际下单和订单管理
"""

import logging
import ccxt
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    REJECTED = "rejected"

@dataclass
class Order:
    """订单数据类"""
    symbol: str
    order_type: OrderType
    side: str  # 'buy' or 'sell'
    amount: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    order_id: Optional[str] = None
    created_time: float = 0
    filled_amount: float = 0

class OrderExecutor:
    """
    订单执行器 - 管理订单生命周期
    """
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        self.logger = logging.getLogger(__name__)
        self.exchange = None
        self.orders: Dict[str, Order] = {}  # order_id -> Order
        self.is_initialized = False
        self.pending_orders = []
        self.positions = {}  # 新增：记录持仓情况
        self.balance = 10000.0  # 新增：起始资金
        # === 新增的功能1：风险检查 ===
    async def risk_check(self, signal):
        """在交易前检查风险"""
        symbol = signal.symbol
        
        # 简单风险规则：单次交易不超过资金的10%
        if signal.signal_type.value == 'buy':
            cost = signal.price * 0.01  # 假设买1%
            if cost > self.balance * 0.1:
                print(f"⛔ 风险检查失败: 交易金额 {cost} 超过资金限制")
                return False
        
        print(f"✅ 风险检查通过")
        return True
    
    # === 新增的功能2：更新持仓 ===
    def update_position(self, symbol, quantity):
        """记录买卖后的持仓变化"""
        if symbol not in self.positions:
            self.positions[symbol] = 0
        
        old_position = self.positions[symbol]
        self.positions[symbol] += quantity
        new_position = self.positions[symbol]
        
        print(f"📈 持仓更新: {symbol} {old_position} -> {new_position}")
        
        # 如果是卖出，资金增加
        if quantity < 0:
            self.balance += abs(quantity) * price
        # 如果是买入，资金减少  
        else:
            self.balance -= quantity * price
            
        print(f"💰 当前资金: {self.balance:.2f}")
    
    async def initialize(self):
        """初始化订单执行器"""
        self.logger.info(f"🔧 初始化订单执行器，交易所: {self.exchange_name}")
        
        # 创建交易所实例（模拟模式）
        exchange_class = getattr(ccxt, self.exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': self.sandbox,  # 使用沙盒环境
            'apiKey': 'YOUR_API_KEY',  # 在实际使用中需要配置
            'secret': 'YOUR_SECRET',
        })
        
        # 加载市场信息
        await self._load_markets()
        self.is_initialized = True
        self.logger.info("✅ 订单执行器初始化完成")
    
    async def _load_markets(self):
        """加载市场信息"""
        try:
            markets = self.exchange.load_markets()
            self.logger.info(f"📋 加载了 {len(markets)} 个交易对的市场信息")
        except Exception as e:
            self.logger.error(f"❌ 加载市场信息失败: {e}")
            raise
    
    async def create_order(self, order: Order) -> Optional[str]:
        """创建订单"""
        if not self.is_initialized:
            self.logger.error("❌ 订单执行器未初始化")
            return None
        
        try:
            self.logger.info(f"💳 创建订单: {order.side} {order.amount} {order.symbol}")
            
            # 在实际交易中，这里会调用交易所API
            # order_result = self.exchange.create_order(
            #     symbol=order.symbol,
            #     type=order.order_type.value,
            #     side=order.side,
            #     amount=order.amount,
            #     price=order.price
            # )
            
            # 模拟订单创建（避免真实交易）
            order.order_id = f"simulated_order_{len(self.orders) + 1}"
            order.status = OrderStatus.OPEN
            order.created_time = self.exchange.milliseconds()
            
            # 存储订单
            self.orders[order.order_id] = order
            
            self.logger.info(f"✅ 订单创建成功: {order.order_id}")
            return order.order_id
            
        except Exception as e:
            self.logger.error(f"❌ 订单创建失败: {e}")
            order.status = OrderStatus.REJECTED
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id not in self.orders:
            self.logger.error(f"❌ 订单不存在: {order_id}")
            return False
        
        try:
            order = self.orders[order_id]
            self.logger.info(f"❌ 取消订单: {order_id}")
            
            # 模拟取消订单
            order.status = OrderStatus.CANCELED
            
            self.logger.info(f"✅ 订单取消成功: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 订单取消失败: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """获取订单状态"""
        if order_id in self.orders:
            return self.orders[order_id].status
        return None
    
    async def get_open_orders(self, symbol: str = None) -> List[Order]:
        """获取未成交订单"""
        open_orders = []
        for order in self.orders.values():
            if order.status == OrderStatus.OPEN:
                if symbol is None or order.symbol == symbol:
                    open_orders.append(order)
        return open_orders
    
    def get_all_orders(self) -> List[Order]:
        """获取所有订单"""
        return list(self.orders.values())

# 模拟交易管理器
class MockOrderExecutor(OrderExecutor):
    """
    模拟订单执行器 - 用于测试，不进行真实交易
    """
    
    async def create_order(self, order: Order) -> Optional[str]:
        """模拟创建订单"""
        self.logger.info(f"🧪 模拟创建订单: {order.side.upper()} {order.amount} {order.symbol}")
        
        order.order_id = f"mock_order_{len(self.orders) + 1}"
        order.status = OrderStatus.OPEN
        order.created_time = self.exchange.milliseconds() if self.exchange else 0
        
        self.orders[order.order_id] = order
        
        # 模拟订单立即成交（测试用）
        if order.order_type == OrderType.MARKET:
            await asyncio.sleep(0.5)  # 模拟网络延迟
            order.status = OrderStatus.CLOSED
            order.filled_amount = order.amount
            self.logger.info(f"✅ 模拟订单立即成交: {order.order_id}")
        
        return order.order_id