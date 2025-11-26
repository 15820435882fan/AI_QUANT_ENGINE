# src/risk/risk_guard.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风险守卫 - 交易风险控制
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class RiskConfig:
    """风险配置"""
    max_position_size: float = 0.1  # 最大仓位10%
    daily_loss_limit: float = 0.05  # 单日最大亏损5%
    max_drawdown: float = 0.15  # 最大回撤15%
    stop_loss: float = 0.02  # 止损2%
    take_profit: float = 0.05  # 止盈5%

class RiskGuard:
    """
    风险守卫 - 实时监控和控制交易风险
    """
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.logger = logging.getLogger(__name__)
        self.positions = {}
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        
    async def check_trading_approval(self, signal, current_price: float) -> bool:
        """检查交易是否被批准"""
        self.logger.info("🔍 风险检查中...")
        
        # 检查仓位限制
        if not await self._check_position_limit(signal):
            self.logger.warning("❌ 交易被拒绝: 超过仓位限制")
            return False
            
        # 检查日亏损限制
        if not await self._check_daily_loss_limit():
            self.logger.warning("❌ 交易被拒绝: 超过日亏损限制")
            return False
            
        # 检查回撤限制
        if not await self._check_drawdown_limit():
            self.logger.warning("❌ 交易被拒绝: 超过最大回撤")
            return False
            
        self.logger.info("✅ 风险检查通过")
        return True
    
    async def _check_position_limit(self, signal) -> bool:
        """检查仓位限制"""
        # 简化实现 - 实际中需要计算总仓位
        current_positions = len(self.positions)
        max_positions = 10  # 最大同时持仓数
        
        if current_positions >= max_positions:
            return False
        return True
    
    async def _check_daily_loss_limit(self) -> bool:
        """检查日亏损限制"""
        if self.daily_pnl <= -self.config.daily_loss_limit:
            return False
        return True
    
    async def _check_drawdown_limit(self) -> bool:
        """检查回撤限制"""
        if self.max_drawdown >= self.config.max_drawdown:
            return False
        return True