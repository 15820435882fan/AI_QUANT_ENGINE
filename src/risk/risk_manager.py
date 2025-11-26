# src/risk/risk_manager.py
#!/usr/bin/env python3
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class RiskConfig:
    """风险配置"""
    max_position_size: float = 0.1  # 单次最大仓位10%
    daily_loss_limit: float = 0.02  # 单日最大亏损2%
    max_drawdown: float = 0.05      # 最大回撤5%
    stop_loss: float = 0.02         # 止损2%
    take_profit: float = 0.04       # 止盈4%

class RiskManager:
    """风险管理系统"""
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.logger = logging.getLogger(__name__)
        self.daily_pnl = 0.0
        self.peak_equity = 0.0
        
    async def validate_trade(self, signal: Dict, current_equity: float, 
                           positions: Dict, today_trades: int) -> Dict:
        """验证交易是否符合风险规则"""
        risk_check = {
            'approved': True,
            'reason': '',
            'adjusted_quantity': None
        }
        
        # 1. 仓位大小检查
        position_size = await self._check_position_size(signal, current_equity)
        if not position_size['approved']:
            return position_size
        
        # 2. 日亏损限制检查
        daily_loss = await self._check_daily_loss(current_equity)
        if not daily_loss['approved']:
            return daily_loss
        
        # 3. 最大回撤检查
        drawdown_check = await self._check_drawdown(current_equity)
        if not drawdown_check['approved']:
            return drawdown_check
        
        # 4. 交易频率检查
        freq_check = await self._check_trading_frequency(today_trades)
        if not freq_check['approved']:
            return freq_check
        
        risk_check['adjusted_quantity'] = position_size['suggested_quantity']
        return risk_check
    
    async def _check_position_size(self, signal: Dict, current_equity: float) -> Dict:
        """检查仓位大小"""
        suggested_quantity = (current_equity * self.config.max_position_size) / signal['price']
        
        return {
            'approved': True,
            'suggested_quantity': suggested_quantity,
            'reason': '仓位检查通过'
        }
    
    async def _check_daily_loss(self, current_equity: float) -> Dict:
        """检查日亏损限制"""
        if self.daily_pnl < -self.config.daily_loss_limit:
            return {
                'approved': False,
                'reason': f'日亏损达到限制: {self.daily_pnl:.2%}'
            }
        return {'approved': True, 'reason': '日亏损检查通过'}
    
    async def _check_drawdown(self, current_equity: float) -> Dict:
        """检查最大回撤"""
        self.peak_equity = max(self.peak_equity, current_equity)
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        if drawdown > self.config.max_drawdown:
            return {
                'approved': False,
                'reason': f'最大回撤超过限制: {drawdown:.2%}'
            }
        return {'approved': True, 'reason': '回撤检查通过'}
    
    async def _check_trading_frequency(self, today_trades: int) -> Dict:
        """检查交易频率"""
        if today_trades > 50:  # 每天最多50笔交易
            return {
                'approved': False,
                'reason': f'交易频率过高: {today_trades}笔'
            }
        return {'approved': True, 'reason': '频率检查通过'}
    
    def update_pnl(self, profit: float):
        """更新盈亏记录"""
        self.daily_pnl += profit