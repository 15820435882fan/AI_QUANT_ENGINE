# account_manager_v31.py

from dataclasses import dataclass, field
from typing import Dict, Any
import math
import time


@dataclass
class AccountState:
    equity: float
    trading_capital: float
    profit_pool: float
    peak_equity: float
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    last_update_ts: float = field(default_factory=time.time)


@dataclass
class WaterfallConfig:
    enable: bool = True
    withdraw_rate: float = 0.10
    growth_threshold: float = 0.01


class AccountManagerV31:
    def __init__(self, initial_equity: float, wf_cfg: WaterfallConfig):
        self.state = AccountState(
            equity=initial_equity,
            trading_capital=initial_equity,
            profit_pool=0.0,
            peak_equity=initial_equity,
        )
        self.wf_cfg = wf_cfg
        self.last_high = initial_equity

    def apply_pnl(self, realized_pnl: float, timestamp: float):
        s = self.state
        s.trading_capital += realized_pnl
        s.total_pnl += realized_pnl

        s.equity = s.trading_capital + s.profit_pool
        s.peak_equity = max(s.peak_equity, s.equity)
        if s.peak_equity > 0:
            dd = (s.equity - s.peak_equity) / s.peak_equity
            s.max_drawdown_pct = min(s.max_drawdown_pct, dd)

        # 抽水逻辑（组合级）
        if self.wf_cfg.enable and s.trading_capital > self.last_high:
            growth = s.trading_capital - self.last_high
            growth_pct = growth / self.last_high
            if growth_pct >= self.wf_cfg.growth_threshold:
                withdraw = growth * self.wf_cfg.withdraw_rate
                s.trading_capital -= withdraw
                s.profit_pool += withdraw
                self.last_high = s.trading_capital
                print(
                    f"[Waterfall] Equity growth {growth_pct*100:.2f}% 抽水 {withdraw:.2f}, "
                    f"交易资金={s.trading_capital:.2f}, 利润池={s.profit_pool:.2f}"
                )

        s.last_update_ts = timestamp

    def get_risk_capital(self) -> float:
        return self.state.trading_capital

    def get_equity(self) -> float:
        return self.state.equity

    def summary(self) -> Dict[str, Any]:
        s = self.state
        return {
            "equity": s.equity,
            "trading_capital": s.trading_capital,
            "profit_pool": s.profit_pool,
            "peak_equity": s.peak_equity,
            "total_pnl": s.total_pnl,
            "max_drawdown_pct": s.max_drawdown_pct,
        }

    def calc_position_size(
        self,
        entry_price: float,
        stop_price: float,
        risk_per_trade: float,
        leverage: float,
        symbol_share: float = 1.0,
    ) -> float:
        """
        简化版仓位计算：
        - risk = equity * risk_per_trade * symbol_share
        - 仓位 = risk / |entry-stop| -> 名义头寸
        - 再用杠杆限制
        返回: 实际下单的币数量（合约张数）
        """
        equity = self.get_risk_capital()
        risk_amount = equity * risk_per_trade * symbol_share
        if risk_amount <= 0 or entry_price <= 0 or stop_price <= 0:
            return 0.0

        stop_dist = abs(entry_price - stop_price)
        if stop_dist <= 0:
            return 0.0

        notional = risk_amount / (stop_dist / entry_price)
        max_notional = equity * leverage * symbol_share
        notional = min(notional, max_notional)

        # 合约张数 = 名义 / 价格
        size = notional / entry_price
        return max(size, 0.0)
