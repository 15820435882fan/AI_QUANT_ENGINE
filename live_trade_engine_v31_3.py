
# -*- coding: utf-8 -*-
"""
live_trade_engine_v31_3.py

V31_3 · 多币轮动 · 实盘引擎（简化版）
- 使用 V31_1 核心策略内核（v31_core_v31_1）
- 支持多币种 TopK 轮动 + 趋势交易
- 当前为「仿真模式」：从本地 5m K线数据中抽取最近一段做一次性决策
  （live_mode=False 时适合作为联调 / 烟雾测试使用）
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

import yaml
import os
# ===== BINANCE 适配器（基于 ccxt） =====
import ccxt

class BinanceClientV31:
    def __init__(self, cfg: dict):
        # 兼容两种写法：binance / exchange
        ex_cfg = cfg.get("binance") or cfg.get("exchange") or {}

        name = ex_cfg.get("name", "binance")
        api_key = ex_cfg.get("apiKey") or ex_cfg.get("api_key")
        secret = ex_cfg.get("secret") or ex_cfg.get("api_secret")

        if not api_key or not secret:
            raise ValueError("Binance 配置缺少 apiKey / secret")

        # 拎出其余参数（enableRateLimit, options 等）
        extra = {
            k: v
            for k, v in ex_cfg.items()
            if k not in ["name", "apiKey", "api_key", "secret", "api_secret"]
        }

        ex_class = getattr(ccxt, name)
        self.ex = ex_class({
            "apiKey": api_key,
            "secret": secret,
            **extra,
        })

    def create_market_order(self, symbol: str, side: str, amount: float, params: dict | None = None):
        """
        side: 'buy' 或 'sell'
        """
        params = params or {}
        print(f"[Binance] 准备下单: {symbol} {side} {amount}")
        return self.ex.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=amount,
            params=params,
        )

    def get_balance(self):
        return self.ex.fetch_balance()


def load_live_config(path: str = "config_live_v31.yaml") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到实盘配置文件: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg

from local_data_engine_v22_9 import LocalDataEngineV22_9
from v31_core_v31_1 import (
    V31CoreConfig,
    WaterfallAccountV31_1,
    build_multi_tf_indicators,
    entry_signal_v31,
    compute_sl_tp_notional_v31,
    Side,
)


@dataclass
class LiveEngineConfig:
    symbols: List[str]
    timeframe: str = "5m"
    days: int = 365
    topk: int = 2
    leverage: float = 10.0
    risk_per_trade: float = 0.01
    initial_equity: float = 10_000.0
    live_mode: bool = False  # True 时可接接交易所 API（当前仅预留接口）


class LiveTradeEngineV31_3:
    def __init__(self, cfg_live: LiveEngineConfig):
        self.cfg_live = cfg_live
        self.engine = LocalDataEngineV22_9()

        # 策略核心配置（与 V31_1 一致）
        self.core_cfg = V31CoreConfig(
            symbol="MULTI",  # 这里只做占位，具体 symbol 在 run_symbol 中传
            days=cfg_live.days,
            leverage=cfg_live.leverage,
            risk_per_trade=cfg_live.risk_per_trade,
        )

        # 账户模型：抽水 + 复利
        self.account = WaterfallAccountV31_1(
            initial_capital=cfg_live.initial_equity,
            withdraw_rate=0.10,
            growth_threshold=0.01,
            enable_waterfall=True,
        )

        # 每个币种的连续亏损计数（如果需要扩展为实盘多次决策）
        self.consecutive_losses: Dict[str, int] = {s: 0 for s in cfg_live.symbols}

    # ========== 趋势评分（用于 TopK 排序） ==========
    def _calc_trend_score(self, df5: pd.DataFrame) -> float:
        """
        简单趋势评分：使用 1H 合成后的趋势强度 + 近期收益率
        这里直接使用 df5 中的 trend_strength / close 做一个粗略 scoring，
        重点在于给出一个可排序的数值，用于 TopK 轮动。
        """
        if len(df5) < 60:
            return 0.0

        # 取最近 N 根 5m 转成大致 1 天、3 天 的收益 + 趋势强度均值
        recent = df5.iloc[-12 * 24:]  # 约 24 小时数据
        if recent.empty:
            return 0.0

        close = recent["close"]
        ret = float(close.iloc[-1] / close.iloc[0] - 1.0)

        strength = float(recent["trend_strength"].mean())
        dir_mean = float(recent["trend_dir"].mean())

        score = ret * 100.0 + strength * 5.0 + dir_mean * 2.0
        return score

    # ========== 主入口：一次性跑一轮 TopK + 信号联调 ==========
    def run_once(self):
        print("==== 启动 V31_3 多币轮动·实盘引擎（仿真一次） ====")
        print(
            f"symbols = {self.cfg_live.symbols}, timeframe={self.cfg_live.timeframe}, "
            f"topk={self.cfg_live.topk}"
        )
        print(
            f"live mode = {self.cfg_live.live_mode}, "
            f"leverage={self.cfg_live.leverage:.1f}x, "
            f"risk_per_trade={self.cfg_live.risk_per_trade*100:.2f}%"
        )

        # 1) 加载并计算所有币种的多周期指标
        df_map: Dict[str, pd.DataFrame] = {}
        score_map: Dict[str, float] = {}

        for sym in self.cfg_live.symbols:
            full_sym = sym.upper()
            if not full_sym.endswith("USDT"):
                full_sym += "USDT"

            print(f"[Ranker] 加载并计算 {full_sym} ...")
            df_raw = self.engine.load_klines(full_sym, self.cfg_live.timeframe, days=self.cfg_live.days)
            ctx = build_multi_tf_indicators(df_raw, self.core_cfg)
            df5 = ctx["df_5m"]
            df_map[full_sym] = df5
            score_map[full_sym] = self._calc_trend_score(df5)

        # 2) TopK 选择
        sorted_syms = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        topk_syms = [s for s, _ in sorted_syms[: self.cfg_live.topk]]

        print(
            f"[TopK] 当前最强: {topk_syms}, "
            f"scores={{{', '.join([f'{k}: {v:.6f}' for k, v in score_map.items()])}}}"
        )

        # 3) 对每个 TopK 币种，根据最后一根 5m K 线给出一次入场信号
        for full_sym in topk_syms:
            df5 = df_map[full_sym]
            if len(df5) < 5:
                continue

            last_ts = df5.index[-1]
            row = df5.iloc[-1]
            prev_row = df5.iloc[-2]

            side, trade_type = entry_signal_v31(
                cfg=self.core_cfg,
                consecutive_losses=self.consecutive_losses.get(full_sym, 0),
                ts=last_ts,
                row=row,
                prev_row=prev_row,
            )

            if side == "FLAT" or trade_type is None:
                continue

            atr_1h = float(row.get("atr_1h", 0.0))
            if atr_1h <= 0:
                continue

            # 入场价（含滑点）
            price_close = float(row["close"])
            if side == "LONG":
                entry_price = price_close * (1.0 + self.core_cfg.slippage)
            else:
                entry_price = price_close * (1.0 - self.core_cfg.slippage)

            equity_for_risk = self.account.risk_capital()
            (
                stop_price,
                take_price,
                notional,
                max_bars,
                sl_abs,
                rr_used,
            ) = compute_sl_tp_notional_v31(
                cfg=self.core_cfg,
                side=side,
                trend_strength=int(row["trend_strength"]),
                entry_price=entry_price,
                atr_1h=atr_1h,
                equity_for_risk=equity_for_risk,
            )

            if notional <= 0 or stop_price <= 0 or take_price <= 0:
                continue

            # 仿真引擎当前只做“信号联调”，不推进持仓循环，pnl=0
            direction_str = "多" if side == "LONG" else "空"
            print(
                f"[Order] 开{direction_str} {full_sym}, "
                f"size={notional/entry_price:.6f}, price≈{entry_price:.4f}, stop={stop_price:.4f}"
            )

        # 4) 打印账户当前状态（仿真模式下未产生实际盈亏）
        equity = self.account.total_equity()
        trading = self.account.risk_capital()
        profit_pool = self.account.profit_pool
        print(
            f"[Account] equity={equity:.2f}, trading={trading:.2f}, "
            f"profit_pool={profit_pool:.2f}"
        )


# ===========================
# CLI 参数解析
# ===========================
def parse_args():
    p = argparse.ArgumentParser(description="V31_3 多币轮动 · 实盘引擎（仿真版）")
    p.add_argument(
        "--symbols",
        type=str,
        default="BTC,ETH,SOL,DOGE",
        help="监控币种列表，用逗号分隔，例如: BTC,ETH,SOL,DOGE",
    )
    p.add_argument("--days", type=int, default=365, help="回测天数 / 历史窗口天数")
    p.add_argument("--topk", type=int, default=2, help="每轮选择最强 TopK 个币种")
    p.add_argument("--leverage", type=float, default=10.0, help="杠杆倍数")
    p.add_argument("--risk-per-trade", type=float, default=0.01, help="单笔风险占比")
    p.add_argument("--initial-equity", type=float, default=10_000.0, help="初始资金")
    p.add_argument("--live", action="store_true", help="启用实盘模式（当前仅保留接口）")
    return p.parse_args()


def main():
    args = parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # 1) 读取配置
    cfg_live = load_live_config("config_live_v31.yaml")

    binance_client = None
    if args.live:          # 只有 --live 才真正下单
        binance_client = BinanceClientV31(cfg_live)

    # 2) 初始化账户（用 V31_1 的账户模型）
    from v31_rule_trend_system_v31_1 import WaterfallAccountV31_1

    initial_equity = cfg_live.get("account", {}).get("initial_equity", 10000.0)
    account = WaterfallAccountV31_1(
        initial_capital=initial_equity,
        withdraw_rate=0.10,
        growth_threshold=0.01,
        enable_waterfall=True,
    )

    # 3) 计算多币种趋势评分，选 TopK（沿用你当前 v31_3 的逻辑）
    #    ... 这里保持你现在的 Ranker 逻辑不动 ...

    # 假设已经选好 topk_syms 和每个符号的 signal（LONG / SHORT / FLAT）
    for sym in topk_syms:
        signal = signal_map[sym]   # 例如 "LONG"

        if signal == "FLAT":
            continue

        # 这里用 V31_1 的 _compute_sl_tp_notional 来算仓位和止损止盈
        # （你选的是方案 A：完整迁移，所以我们会共用那一套逻辑）

        # 假设我们已经有 entry_price、atr_1h、trend_strength、account.risk_capital()
        stop_price, take_price, notional, max_bars_hold, sl_abs, rr_used = core.compute_sl_tp_notional_for_live(
            side=signal,
            trend_strength=trend_strength_now,
            entry_price=entry_price,
            atr_1h=atr_1h,
            risk_equity=account.risk_capital(),
        )

        # 打印仿真
        print(f"[Order] {signal} {sym}, notional={notional:.2f}, entry≈{entry_price:.4f}, "
              f"SL={stop_price:.4f}, TP={take_price:.4f}, RR={rr_used:.2f}")

        # 4) 如果 --live，则通过 binance_client 实际下单
        if binance_client is not None and notional > 0:
            # 这里要把 notional 转成交易所的 amount：
            # amount = notional / entry_price / leverage 或按你的合约规则调
            amount = notional / entry_price / args.leverage

            side = "buy" if signal == "LONG" else "sell"
            binance_client.create_market_order(sym, side, amount)

    )
    engine = LiveTradeEngineV31_3(cfg_live)
    engine.run_once()


if __name__ == "__main__":
    main()
