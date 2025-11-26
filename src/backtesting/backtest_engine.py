# src/backtesting/backtest_engine.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测引擎 - 历史数据测试和策略验证
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import time

# 🔧 修复导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.data.data_pipeline import MarketData, DataType
from src.strategies.strategy_orchestrator import TradingSignal, SignalType

@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 10000.0
    start_date: str = "2024-01-01"
    end_date: str = "2024-03-01"
    commission: float = 0.001  # 交易手续费
    slippage: float = 0.0005   # 滑点

@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float
    final_balance: float
    
    # 详细交易记录
    trades: List[Dict]
    equity_curve: pd.DataFrame

class BacktestEngine:
    """
    回测引擎 - 在历史数据上测试交易策略
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.logger = logging.getLogger(__name__)
        
        # 回测状态
        self.current_date = None
        self.balance = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
    async def run_backtest(self, strategy, historical_data: pd.DataFrame) -> BacktestResult:
        """运行回测"""
        self.logger.info("🚀 开始回测...")
        
        # 初始化状态
        self._initialize_backtest()
        
        # 确保数据有正确的时间索引
        if not isinstance(historical_data.index, pd.DatetimeIndex):
            self.logger.warning("⚠️ 数据没有时间索引，使用顺序索引")
        
        # 按时间顺序处理历史数据
        for idx, row in historical_data.iterrows():
            self.current_date = idx
            
            # 更新当前价格
            current_price = row['close']
            
            # 执行策略分析
            market_data = self._create_market_data(row, idx)
            signal = await strategy.analyze(market_data)
            
            # 处理交易信号
            if signal:
                await self._execute_trade(signal, current_price, idx)
            
            # 记录权益曲线
            self._record_equity(current_price)
            
        # 计算回测结果
        result = self._calculate_results()
        self.logger.info("✅ 回测完成")
        
        return result
    
    def _initialize_backtest(self):
        """初始化回测状态"""
        self.balance = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
    
    def _create_market_data(self, row: pd.Series, timestamp) -> MarketData:
        """创建市场数据对象"""
        # 🔧 修复时间戳处理
        if hasattr(timestamp, 'timestamp'):
            # 如果是时间对象
            timestamp_value = timestamp.timestamp()
        elif hasattr(timestamp, 'to_pydatetime'):
            # 如果是pandas时间戳
            timestamp_value = timestamp.to_pydatetime().timestamp()
        else:
            # 使用当前时间
            timestamp_value = time.time()
        
        return MarketData(
            symbol='BTC/USDT',
            data_type=DataType.OHLCV,
            data={
                'open': float(row['open']),
                'high': float(row['high']), 
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row.get('volume', 0))
            },
            timestamp=timestamp_value
        )
    
    async def _execute_trade(self, signal, current_price: float, timestamp):
        """执行交易"""
        symbol = signal.symbol
        
        # 风险检查
        if not await self._risk_check(signal, current_price):
            return
        
        # 计算交易数量
        quantity = self._calculate_position_size(signal, current_price)
        
        # 记录交易
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'signal_type': signal.signal_type.value,
            'price': current_price,
            'quantity': quantity,
            'commission': abs(quantity * current_price * self.config.commission)
        }
        
        # 更新资金和持仓
        if signal.signal_type.value == 'buy':
            cost = quantity * current_price + trade['commission']
            if cost <= self.balance:
                self.balance -= cost
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                trade['status'] = 'executed'
                self.logger.info(f"💰 买入 {quantity:.4f} {symbol} @ {current_price:.2f}")
            else:
                trade['status'] = 'rejected_insufficient_balance'
                self.logger.warning(f"⛔ 资金不足，无法买入 {symbol}")
        else:  # sell
            current_position = self.positions.get(symbol, 0)
            if quantity <= current_position:
                self.positions[symbol] = current_position - quantity
                self.balance += quantity * current_price - trade['commission']
                trade['status'] = 'executed'
                self.logger.info(f"💰 卖出 {quantity:.4f} {symbol} @ {current_price:.2f}")
            else:
                trade['status'] = 'rejected_insufficient_position'
                self.logger.warning(f"⛔ 持仓不足，无法卖出 {symbol}")
        
        self.trades.append(trade)
    
    async def _risk_check(self, signal, current_price: float) -> bool:
        """风险检查"""
        # 基础风险规则
        if signal.signal_type.value == 'buy':
            # 单次交易不超过资金的20%
            position_value = self._calculate_position_size(signal, current_price) * current_price
            if position_value > self.balance * 0.2:
                self.logger.warning("⛔ 风险检查失败: 交易金额超过限制")
                return False
        return True
    
    def _calculate_position_size(self, signal, current_price: float) -> float:
        """计算头寸大小"""
        if signal.signal_type.value == 'buy':
            # 使用5%的资金
            risk_capital = self.balance * 0.05
            return risk_capital / current_price
        else:
            # 卖出当前持仓的50%
            return self.positions.get(signal.symbol, 0) * 0.5
    
    def _record_equity(self, current_price: float):
        """记录权益曲线"""
        position_value = sum(
            quantity * current_price 
            for symbol, quantity in self.positions.items()
        )
        total_equity = self.balance + position_value
        self.equity_curve.append({
            'timestamp': self.current_date,
            'equity': total_equity,
            'balance': self.balance,
            'position_value': position_value
        })
    
    def _calculate_results(self) -> BacktestResult:
        """计算回测结果指标"""
        if not self.equity_curve:
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, self.balance, [], pd.DataFrame())
        
        equity_df = pd.DataFrame(self.equity_curve)
        if 'timestamp' in equity_df.columns:
            equity_df.set_index('timestamp', inplace=True)
        
        # 计算关键指标
        initial_equity = equity_df['equity'].iloc[0]
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - initial_equity) / initial_equity
        
        # 年化收益（简化计算）
        if len(equity_df) > 1:
            days = 10  # 假设10天
            annual_return = (1 + total_return) ** (365 / days) - 1
        else:
            annual_return = 0
        
        # 最大回撤
        equity_series = equity_df['equity']
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        # 夏普比率（简化）
        daily_returns = equity_df['equity'].pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(365)
        else:
            sharpe_ratio = 0
        
        # 胜率（简化）
        executed_trades = [t for t in self.trades if t.get('status') == 'executed']
        win_rate = 0.5  # 临时值
        
        # 盈利因子
        profit_factor = 1.0  # 临时值
        
        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(executed_trades),
            profit_factor=profit_factor,
            final_balance=final_equity,
            trades=executed_trades,
            equity_curve=equity_df
        )

# 数据获取模块
class DataManager:
    """历史数据管理器"""
    
    @staticmethod
    async def load_historical_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """加载历史数据"""
        # 生成带正确时间戳的数据
        dates = pd.date_range(start=start_date, end=end_date, freq='1min')
        np.random.seed(42)
        
        data = []
        price = 50000.0
        
        for date in dates:
            # 模拟价格波动
            change = np.random.normal(0, 0.002)
            price = max(price * (1 + change), 1000)
            
            data.append({
                'timestamp': date,
                'open': float(price * (1 + np.random.normal(0, 0.001))),
                'high': float(price * (1 + abs(np.random.normal(0, 0.002)))),
                'low': float(price * (1 - abs(np.random.normal(0, 0.002)))),
                'close': float(price),
                'volume': float(np.random.uniform(1000, 5000))
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)  # 设置时间戳为索引
        return df