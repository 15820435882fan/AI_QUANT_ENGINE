# -*- coding: utf-8 -*-
"""
live_trade_engine_v31_13.py

V31_13 · 多币轮动 · Binance USDT-M 永续合约 实盘引擎

在 V31_12 基础上新增：

1）启动后自动接管 Binance 当前未平仓持仓（USDT-M，且在本次监控的 symbols 列表中）：
    - 通过 fetch_balance(params={"type":"future"}) 读取 info.positions
    - 对每个 positionAmt != 0 且 symbol 在监控列表中的仓位：
        * 判定 side = LONG / SHORT
        * 使用 entryPrice × abs(positionAmt) 作为 notional
        * 从本地 df_map[symbol] 中提取 atr_1h, trend_strength
        * 调用 compute_sl_tp_notional_v31 获得 stop_price / take_price / max_bars
        * 写入 engine.positions[symbol]，之后纳入自动平仓逻辑管理

2）仍然保留：
    - hedge_mode 自动/手动检测（单向/对冲持仓）
    - V31_1 风格的趋势信号 + ATR 止损 + RR 止盈
    - 多币 TopK 轮动 + 单币单仓位
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional
import time
import os
import sys

import yaml
import ccxt
import pandas as pd

from v31_core_v31_1 import (
    V31CoreConfig,
    WaterfallAccountV31_1,
    build_multi_tf_indicators,
    entry_signal_v31,
    compute_sl_tp_notional_v31,
)


# ==============================
# 配置加载
# ==============================
def load_live_config(path: str = "config_live_v31.yaml") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到实盘配置文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


# ==============================
# Binance USDT-M Futures 适配器
# ==============================
class BinanceFuturesV31:
    def __init__(self, cfg: dict):
        """
        期望 YAML 结构示例：

        binance:
          name: binance
          apiKey: "..."
          secret: "..."
          enable_trading: true
          hedge_mode: true   # 可选，若写了则强制使用
          enableRateLimit: true
          options:
            defaultType: future
            adjustForTimeDifference: true
          proxies:
            http: null
            https: null
        """
        ex_cfg = cfg.get("binance") or cfg.get("exchange") or {}

        name = ex_cfg.get("name", "binance")
        api_key = ex_cfg.get("apiKey") or ex_cfg.get("api_key")
        secret = ex_cfg.get("secret") or ex_cfg.get("api_secret")
        if not api_key or not secret:
            raise ValueError("Binance 配置缺少 apiKey / secret")

        # 是否真实下单
        self.enable_trading: bool = bool(ex_cfg.get("enable_trading", False))

        # 拆出 proxies 和 options，其余交给 ccxt
        proxies = ex_cfg.get("proxies") or None
        options = ex_cfg.get("options") or {}

        other = {
            k: v
            for k, v in ex_cfg.items()
            if k
            not in [
                "name",
                "apiKey",
                "api_key",
                "secret",
                "api_secret",
                "enable_trading",
                "proxies",
                "options",
                "hedge_mode",
            ]
        }

        ex_class = getattr(ccxt, name)
        self.ex = ex_class(
            {
                "apiKey": api_key,
                "secret": secret,
                "enableRateLimit": True,
                "proxies": proxies,
                "options": options,
                **other,
            }
        )

        # 1) 先看 YAML 是否显式指定 hedge_mode
        explicit_hedge = ex_cfg.get("hedge_mode")
        if explicit_hedge is not None:
            self.hedge_mode = bool(explicit_hedge)
            mode_str = "对冲模式(双向持仓)" if self.hedge_mode else "单向持仓模式"
            print(f"[Binance] 配置文件中显式指定 hedge_mode={self.hedge_mode} → {mode_str}")
            return

        # 2) 否则尝试通过 fetch_balance 判断
        self.hedge_mode: bool = False
        try:
            bal = self.ex.fetch_balance(params={"type": "future"})
            info = bal.get("info") or {}
            positions = info.get("positions") or []
            sides = set()
            for p in positions:
                side = p.get("positionSide")
                if side:
                    sides.add(str(side).upper())

            if not sides:
                # 没检测到 positionSide 字段，保守当单向持仓
                self.hedge_mode = False
                print(
                    "[Binance] 无 positionSide 字段信息，默认按单向持仓模式处理 "
                    f"(sides={sides})"
                )
            else:
                # 典型：
                #   对冲模式：出现 LONG / SHORT
                #   单向模式：只有 BOTH
                if "LONG" in sides or "SHORT" in sides:
                    self.hedge_mode = True
                    print(
                        "[Binance] 自动检测：账户为对冲模式(双向持仓)，"
                        f"positionSide 集合={sides}"
                    )
                else:
                    self.hedge_mode = False
                    print(
                        "[Binance] 自动检测：账户为单向持仓模式，"
                        f"positionSide 集合={sides}"
                    )
        except Exception as e:
            # 无法探测时，保守认定为单向模式
            self.hedge_mode = False
            print(f"[Binance] 警告：无法自动检测持仓模式，默认按单向持仓模式处理。err={e}")

    # ---- 行情 ----
    def fetch_ohlcv_5m(self, symbol: str, limit: int = 500) -> Optional[pd.DataFrame]:
        """
        固定拉取最近 limit 根 5m K 线，默认 500 根。
        不再用 since，避免只拿到 1 根 K 线的情况。
        """
        try:
            ohlcv = self.ex.fetch_ohlcv(symbol, timeframe="5m", limit=limit)
        except Exception as e:
            print(f"[Binance] fetch_ohlcv 失败: symbol={symbol}, err={e}")
            return None

        if not ohlcv:
            print(f"[Binance] fetch_ohlcv 返回空数据: symbol={symbol}")
            return None

        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        print(f"[Binance] {symbol} 5m K 线获取成功，bars={len(df)}")
        return df

    # ---- 账户（可选）----
    def fetch_futures_equity_usdt(self) -> Optional[float]:
        """
        获取 USDT-M 永续账户 USDT 权益（仅用于初始化初始资金，不参与资金演化）
        """
        try:
            balance = self.ex.fetch_balance(params={"type": "future"})
        except Exception as e:
            print(f"[Binance] 获取期货账户余额失败: {e}")
            try:
                balance = self.ex.fetch_balance()
            except Exception as e2:
                print(f"[Binance] 退回普通 fetch_balance 也失败: {e2}")
                return None

        eq = None
        total = balance.get("total")
        if isinstance(total, dict):
            eq = total.get("USDT")

        if eq is None and "USDT" in balance:
            usdt_info = balance["USDT"]
            if isinstance(usdt_info, dict):
                eq = usdt_info.get("total") or usdt_info.get("free")

        if eq is None:
            print("[Binance] 警告：未能解析出 USDT 总权益，原始结构：")
            print(balance)
            return None

        return float(eq)

    # ---- 下单 ----
    def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        params: Optional[dict] = None,
    ):
        """
        side: "buy" / "sell"
        amount: 合约数量（张数）
        params: 可包含 positionSide / reduceOnly 等
        """
        params = params or {}
        if amount <= 0:
            print(f"[Binance] 下单数量为 0，跳过: {symbol} {side} {amount}")
            return None

        if not self.enable_trading:
            print(f"[Binance] enable_trading=False，跳过真实下单: {symbol} {side} {amount}")
            return None

        print(f"[Binance] 准备下单: {symbol} {side} {amount}, params={params}")
        try:
            order = self.ex.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=amount,
                params=params,
            )
            print(f"[Binance] 下单成功: {order}")
            return order
        except Exception as e:
            print(f"[Binance] 下单失败: {e}")
            return None


# ==============================
# 引擎配置
# ==============================
@dataclass
class LiveEngineConfig:
    symbols: List[str]
    timeframe: str = "5m"
    topk: int = 2
    leverage: float = 3.0        # 默认用回测 V31_1 的 3x
    risk_per_trade: float = 0.01
    refresh_seconds: int = 10
    initial_equity: float = 10_000.0
    live_mode: bool = False
    # 止盈止损参数（与 V31_1 核心一致，可通过命令行微调）
    rr_strong: float = 4.0
    rr_normal: float = 3.0
    sl_mult_strong: float = 3.5
    sl_mult_normal: float = 3.0


# ==============================
# 多币轮动实盘引擎
# ==============================

@dataclass
class LivePositionV31:
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_ts: pd.Timestamp
    entry_price: float
    qty: float          # 交易所下单使用的数量（contracts）
    notional: float     # 名义仓位（用于资金与 PnL 计算）
    stop_price: float
    take_price: float
    max_bars: int
    bars_held: int = 0
    last_bar_ts: Optional[pd.Timestamp] = None


class LiveTradeEngineV31_13:
    def __init__(
        self,
        cfg_live: LiveEngineConfig,
        binance: Optional[BinanceFuturesV31] = None,
    ):
        self.cfg_live = cfg_live
        self.binance = binance

        # 与回测核心保持一致
        self.core_cfg = V31CoreConfig(
            symbol="MULTI",
            days=365,
            leverage=cfg_live.leverage,
            risk_per_trade=cfg_live.risk_per_trade,
        )

        # 将实盘参数同步到核心配置（RR / ATR 止损倍数）
        self.core_cfg.rr_strong = cfg_live.rr_strong
        self.core_cfg.rr_normal = cfg_live.rr_normal
        self.core_cfg.sl_mult_strong = cfg_live.sl_mult_strong
        self.core_cfg.sl_mult_normal = cfg_live.sl_mult_normal

        # 账户：真实复利，不抽水
        self.account = WaterfallAccountV31_1(
            initial_capital=cfg_live.initial_equity,
            withdraw_rate=0.0,
            growth_threshold=0.01,
            enable_waterfall=False,
        )

        # 若在实盘模式下，可以用 Binance 实盘权益作为“初始资金”
        if self.cfg_live.live_mode and self.binance is not None:
            eq = self.binance.fetch_futures_equity_usdt()
            if eq is not None and eq > 0:
                self.account.initial_capital = float(eq)
                self.account.trading_capital = float(eq)
                self.account.last_high = float(eq)
                print(f"[Account] 使用 Binance 实盘 USDT 权益作为初始资金: {eq:.2f}")
            else:
                print(
                    f"[Account] 未能获取实盘权益，继续使用配置中的 initial_equity={cfg_live.initial_equity:.2f}"
                )

        # 资金演化完全由策略 PnL 决定
        self.consecutive_losses: Dict[str, int] = {s: 0 for s in cfg_live.symbols}

        # 实盘持仓跟踪（仅记录由策略管理的持仓）
        self.positions: Dict[str, LivePositionV31] = {}

        # 是否已经从交易所同步过初始持仓
        self.synced_positions_from_exchange: bool = False

        # 运行统计
        self.start_time = pd.Timestamp.utcnow()
        self.refresh_count: int = 0
        self.total_signal_count: int = 0
        self.total_order_sent: int = 0
        self.total_order_success: int = 0
        self.total_order_fail: int = 0

    # ---------- 趋势评分 ----------
    def _calc_trend_score(self, df5: pd.DataFrame) -> float:
        if df5 is None or df5.empty or len(df5) < 60:
            return 0.0

        recent = df5.iloc[-12 * 24 :]  # 约 24 小时
        if recent.empty:
            return 0.0

        close = recent["close"]
        ret = float(close.iloc[-1] / close.iloc[0] - 1.0)

        strength = float(recent["trend_strength"].mean())
        dir_mean = float(recent["trend_dir"].mean())

        score = ret * 100.0 + strength * 5.0 + dir_mean * 2.0
        return score

    # ---------- 启动后从交易所同步当前未平仓位 ----------
    def _sync_positions_from_exchange(self, df_map: Dict[str, pd.DataFrame]):
        """
        仅在 live_mode 且尚未同步时执行一次：
        - 从 Binance 期货账户中读取当前所有未平仓位
        - 对于 symbol 在本次监控列表中的仓位，生成 LivePositionV31 写入 self.positions
        """
        if self.synced_positions_from_exchange:
            return
        if not self.cfg_live.live_mode or self.binance is None:
            return

        try:
            bal = self.binance.ex.fetch_balance(params={"type": "future"})
        except Exception as e:
            print(f"[Sync] 获取期货账户持仓失败，无法同步已有仓位: {e}")
            self.synced_positions_from_exchange = True
            return

        info = bal.get("info") or {}
        positions = info.get("positions") or []
        if not positions:
            print("[Sync] 期货账户无持仓记录可同步。")
            self.synced_positions_from_exchange = True
            return

        symbols_set = {s.upper() if s.upper().endswith("USDT") else (s.upper() + "USDT")
                       for s in self.cfg_live.symbols}

        synced_count = 0
        for p in positions:
            symbol = str(p.get("symbol") or "").upper()
            if symbol not in symbols_set:
                continue

            pos_amt = float(p.get("positionAmt") or 0.0)
            if abs(pos_amt) < 1e-8:
                continue

            entry_price = float(p.get("entryPrice") or 0.0)
            if entry_price <= 0:
                continue

            side = "LONG" if pos_amt > 0 else "SHORT"
            qty = abs(pos_amt)
            notional = qty * entry_price

            df5 = df_map.get(symbol)
            if df5 is None or len(df5) < 5:
                print(f"[Sync] {symbol} 虽有持仓，但本地 K 线不足，暂不接管。")
                continue

            last_ts = df5.index[-1]
            row = df5.iloc[-1]

            atr_1h = float(row.get("atr_1h", 0.0))
            if atr_1h <= 0:
                print(f"[Sync] {symbol} 无有效 atr_1h，暂不接管。")
                continue

            trend_strength = int(row.get("trend_strength", 0))

            # 使用当前参数重新计算一套 stop / take / max_bars
            try:
                stop_price, take_price, _, max_bars, _, rr_used = compute_sl_tp_notional_v31(
                    cfg=self.core_cfg,
                    side=side,
                    trend_strength=trend_strength,
                    entry_price=entry_price,
                    atr_1h=atr_1h,
                    equity_for_risk=self.account.risk_capital(),  # 仅用于内部 sizing，我们只取 stop/take/max_bars
                )
            except Exception as e:
                print(f"[Sync] 计算 {symbol} 的 SL/TP 失败，暂不接管: {e}")
                continue

            if stop_price <= 0 or take_price <= 0 or max_bars <= 0:
                print(f"[Sync] {symbol} 计算出的 SL/TP 非法，暂不接管。")
                continue

            self.positions[symbol] = LivePositionV31(
                symbol=symbol,
                side=side,
                entry_ts=last_ts,
                entry_price=entry_price,
                qty=qty,
                notional=notional,
                stop_price=stop_price,
                take_price=take_price,
                max_bars=int(max_bars),
                bars_held=0,
                last_bar_ts=None,
            )

            synced_count += 1
            print(
                f"[Sync] 接管已有实盘持仓: {symbol}, side={side}, qty={qty}, "
                f"entry={entry_price:.4f}, SL={stop_price:.4f}, TP={take_price:.4f}, "
                f"max_bars={int(max_bars)}, notional={notional:.2f}, RR_used={rr_used:.2f}"
            )

        if synced_count == 0:
            print("[Sync] 未发现需要接管的实盘持仓。")
        else:
            print(f"[Sync] 本次共接管 {synced_count} 个实盘持仓。")

        self.synced_positions_from_exchange = True

    # ---------- 自动平仓逻辑 ----------
    def _handle_auto_exit(self, df_map: Dict[str, pd.DataFrame]):
        to_close = []

        for sym, pos in list(self.positions.items()):
            df5 = df_map.get(sym)
            if df5 is None or len(df5) < 5:
                continue

            last_ts = df5.index[-1]
            row = df5.iloc[-1]
            prev_row = df5.iloc[-2]

            # 更新持仓 bar 计数（仅当出现新 K 线时才 +1）
            if pos.last_bar_ts is None or last_ts > pos.last_bar_ts:
                pos.bars_held += 1
                pos.last_bar_ts = last_ts

            reason = None
            exit_price = None

            # 1) 最大持仓 bars 约束
            if pos.max_bars > 0 and pos.bars_held >= pos.max_bars:
                reason = f"持仓 bars 超限 ({pos.bars_held} >= {pos.max_bars})"
                exit_price = float(row["close"])

            # 2) 基于 K 线高低价的 TP / SL 判定
            high = float(row["high"])
            low = float(row["low"])
            if reason is None:
                if pos.side == "LONG":
                    if low <= pos.stop_price:
                        reason = f"触发止损 {pos.stop_price:.4f}"
                        exit_price = pos.stop_price
                    elif high >= pos.take_price:
                        reason = f"触发止盈 {pos.take_price:.4f}"
                        exit_price = pos.take_price
                else:  # SHORT
                    if high >= pos.stop_price:
                        reason = f"触发止损 {pos.stop_price:.4f}"
                        exit_price = pos.stop_price
                    elif low <= pos.take_price:
                        reason = f"触发止盈 {pos.take_price:.4f}"
                        exit_price = pos.take_price

            # 3) 趋势反转
            if reason is None:
                side_new, trade_type_new = entry_signal_v31(
                    cfg=self.core_cfg,
                    consecutive_losses=self.consecutive_losses.get(sym, 0),
                    ts=last_ts,
                    row=row,
                    prev_row=prev_row,
                )
                if side_new != "FLAT" and side_new != pos.side:
                    reason = f"趋势反转: {pos.side} -> {side_new}"
                    exit_price = float(row["close"])

            if reason is None or exit_price is None:
                continue

            print(f"[Exit] {sym} @ {last_ts}, side={pos.side}, reason={reason}, exit_price={exit_price:.4f}")

            # === 先计算 PnL，更新账户 ===
            notional = pos.notional
            if notional > 0:
                if pos.side == "LONG":
                    price_change_pct = (exit_price - pos.entry_price) / pos.entry_price
                else:
                    price_change_pct = (pos.entry_price - exit_price) / pos.entry_price

                gross_pnl = notional * price_change_pct
                fee_exit = notional * self.core_cfg.fee_rate
                slippage_cost = notional * self.core_cfg.slippage
                pnl = gross_pnl - fee_exit - slippage_cost

                try:
                    self.account.apply_pnl(pnl, last_ts)
                except Exception as e:
                    print(f"[Account] 应用平仓 PnL 时出错: {e}")

                if pnl <= 0:
                    self.consecutive_losses[sym] = self.consecutive_losses.get(sym, 0) + 1
                else:
                    self.consecutive_losses[sym] = 0

                print(
                    f"[PnL] {sym} side={pos.side}, notional={notional:.2f}, "
                    f"entry={pos.entry_price:.4f}, exit={exit_price:.4f}, pnl={pnl:.2f}"
                )

            # === 实盘下平仓单 ===
            if self.cfg_live.live_mode and self.binance is not None and pos.qty > 0:
                side_str = "sell" if pos.side == "LONG" else "buy"
                close_params: Dict[str, object] = {
                    "reduceOnly": True,
                }
                if self.binance.hedge_mode:
                    close_params["positionSide"] = "LONG" if pos.side == "LONG" else "SHORT"

                self.total_order_sent += 1
                try:
                    order = self.binance.create_market_order(
                        symbol=sym,
                        side=side_str,
                        amount=pos.qty,
                        params=close_params,
                    )
                except Exception as e:
                    print(f"[ExitOrder] {sym} 平仓下单异常: {e}")
                    order = None

                if order is not None:
                    self.total_order_success += 1
                    print(f"[ExitOrder] {sym} 平仓下单成功。")
                else:
                    self.total_order_fail += 1
                    print(f"[ExitOrder] {sym} 平仓下单失败。")

            to_close.append(sym)

        for sym in to_close:
            self.positions.pop(sym, None)

    # ---------- 单轮计算 ----------
    def run_one_cycle(self):
        self.refresh_count += 1
        cycle_signal_count = 0
        cycle_order_sent = 0
        cycle_order_success = 0
        cycle_order_fail = 0

        print(
            f"\n==== V31_13 周期刷新 · symbols={self.cfg_live.symbols}, "
            f"topk={self.cfg_live.topk}, live={self.cfg_live.live_mode}, "
            f"refresh={self.cfg_live.refresh_seconds}s ===="
        )

        if self.cfg_live.live_mode and self.binance is None:
            print("[Engine] live_mode=True 但未提供 Binance 客户端，直接返回。")
            return

        df_map: Dict[str, pd.DataFrame] = {}
        score_map: Dict[str, float] = {}

        # 1) 拉多币种 5m K 线 & 指标
        for sym in self.cfg_live.symbols:
            full_sym = sym.upper()
            if not full_sym.endswith("USDT"):
                full_sym += "USDT"

            df_raw = None
            if self.binance is not None:
                df_raw = self.binance.fetch_ohlcv_5m(full_sym, limit=500)

            if df_raw is None or df_raw.empty:
                print(f"[Engine] 无法获取 {full_sym} 的实时 K 线，跳过。")
                continue

            ctx = build_multi_tf_indicators(df_raw, self.core_cfg)
            df5 = ctx["df_5m"].copy()

            ind_cols = [
                c
                for c in df5.columns
                if c
                not in [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            ]
            df5[ind_cols] = df5[ind_cols].ffill()

            df_map[full_sym] = df5
            score_map[full_sym] = self._calc_trend_score(df5)

        if not score_map:
            print("[Engine] 本轮未能获得任何币种的评分，结束本轮。")
            return

        sorted_syms = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        topk_syms = [s for s, _ in sorted_syms[: self.cfg_live.topk]]
        scores_str = ", ".join([f"{k}: {v:.6f}" for k, v in score_map.items()])
        print(f"[TopK] 当前最强: {topk_syms}, scores={{ {scores_str} }}")

        # 2) 若尚未从交易所同步实盘持仓，则先同步一次
        self._sync_positions_from_exchange(df_map)

        # 3) 对现有持仓执行自动平仓
        self._handle_auto_exit(df_map)

        # 4) TopK → 入场信号（仅对当前无持仓的币种）
        for full_sym in topk_syms:
            df5 = df_map.get(full_sym)
            if df5 is None or len(df5) < 5:
                continue

            # 若已持仓则不重复开仓
            if full_sym in self.positions:
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

            if notional <= 0 or stop_price <= 0 or take_price <= 0 or max_bars <= 0:
                continue

            qty = notional / entry_price
            direction_str = "多" if side == "LONG" else "空"
            print(
                f"[Signal] {full_sym} @ {last_ts}, 开{direction_str}, "
                f"notional={notional:.2f}, qty={qty:.6f}, entry≈{entry_price:.4f}, "
                f"SL={stop_price:.4f}, TP={take_price:.4f}, RR={rr_used:.2f}"
            )

            self.total_signal_count += 1
            cycle_signal_count += 1

            # 入场手续费
            fee_entry = notional * self.core_cfg.fee_rate
            try:
                self.account.apply_pnl(-fee_entry, last_ts)
            except Exception as e:
                print(f"[Account] 应用开仓手续费时出错: {e}")

            if self.cfg_live.live_mode and self.binance is not None:
                side_str = "buy" if side == "LONG" else "sell"
                amount = qty

                open_params: Dict[str, object] = {}
                if self.binance.hedge_mode:
                    open_params["positionSide"] = "LONG" if side == "LONG" else "SHORT"

                self.total_order_sent += 1
                cycle_order_sent += 1
                order = self.binance.create_market_order(
                    symbol=full_sym,
                    side=side_str,
                    amount=amount,
                    params=open_params,
                )
                if order is not None:
                    self.total_order_success += 1
                    cycle_order_success += 1
                    self.positions[full_sym] = LivePositionV31(
                        symbol=full_sym,
                        side=side,
                        entry_ts=last_ts,
                        entry_price=entry_price,
                        qty=amount,
                        notional=notional,
                        stop_price=stop_price,
                        take_price=take_price,
                        max_bars=int(max_bars),
                        bars_held=0,
                        last_bar_ts=None,
                    )
                else:
                    self.total_order_fail += 1
                    cycle_order_fail += 1
            else:
                # 非实盘：只在模型层记录持仓
                self.positions[full_sym] = LivePositionV31(
                    symbol=full_sym,
                    side=side,
                    entry_ts=last_ts,
                    entry_price=entry_price,
                    qty=qty,
                    notional=notional,
                    stop_price=stop_price,
                    take_price=take_price,
                    max_bars=int(max_bars),
                    bars_held=0,
                    last_bar_ts=None,
                )

        # 5) 账户打印
        equity = self.account.total_equity()
        trading = self.account.risk_capital()
        profit_pool = self.account.profit_pool
        print(
            f"[Account] 模型层资金: equity={equity:.2f}, trading={trading:.2f}, "
            f"profit_pool={profit_pool:.2f}"
        )

        if self.cfg_live.live_mode and self.binance is not None:
            eq_real = self.binance.fetch_futures_equity_usdt()
            if eq_real is not None:
                print(f"[Account] Binance 实盘 USDT 权益(仅供参考): {eq_real:.2f}")

        elapsed_min = (pd.Timestamp.utcnow() - self.start_time).total_seconds() / 60.0
        print("\n==== [Engine Status · V31_13] ============================")
        if self.start_time.tzinfo:
            start_str = self.start_time.tz_convert("Asia/Shanghai")
        else:
            start_str = self.start_time
        print(f"启动时间: {start_str}")
        print(f"运行时长: {elapsed_min:.2f} 分钟")
        print(f"刷新次数: {self.refresh_count}")
        print(f"本轮信号数: {cycle_signal_count}, 累计信号数: {self.total_signal_count}")
        print(
            f"本轮下单数: {cycle_order_sent}, 成功: {cycle_order_success}, 失败: {cycle_order_fail}"
        )
        print(
            f"累计下单数: {self.total_order_sent}, 成功: {self.total_order_success}, 失败: {self.total_order_fail}"
        )
        print("=========================================================")

    # ---------- 主循环 ----------
    def run_forever(self):
        print("==== 启动 V31_13 多币轮动·Binance USDT-M Futures 实盘引擎（实时循环） ====")
        print(
            f"symbols = {self.cfg_live.symbols}, timeframe={self.cfg_live.timeframe}, "
            f"topk={self.cfg_live.topk}"
        )
        print(
            f"live mode = {self.cfg_live.live_mode}, leverage={self.cfg_live.leverage:.1f}x, "
            f"risk_per_trade={self.cfg_live.risk_per_trade*100:.2f}%, "
            f"refresh={self.cfg_live.refresh_seconds}s"
        )

        while True:
            try:
                self.run_one_cycle()
            except KeyboardInterrupt:
                print("\n[Engine] 收到手动中断信号，准备退出...")
                break
            except Exception as e:
                print(f"[Engine] 本轮出现异常: {e}", file=sys.stderr)
            time.sleep(self.cfg_live.refresh_seconds)


# ===========================
# CLI 参数解析 & main
# ===========================
def parse_args():
    p = argparse.ArgumentParser(description="V31_13 多币轮动 · Binance USDT-M Futures 实时引擎")
    p.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT",
        help="监控币种列表，用逗号分隔，例如: BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT",
    )
    p.add_argument("--topk", type=int, default=2, help="每轮选择最强 TopK 个币种")
    p.add_argument(
        "--leverage",
        type=float,
        default=3.0,
        help="杠杆倍数（默认 3.0，与 V31_1 回测配置一致）",
    )
    p.add_argument(
        "--risk-per-trade",
        type=float,
        default=0.01,
        help="单笔风险占比，例如 0.01 = 1%",
    )
    p.add_argument(
        "--rr-strong",
        type=float,
        default=4.0,
        help="强趋势 RR（默认 4.0，与 V31_1 回测一致）",
    )
    p.add_argument(
        "--rr-normal",
        type=float,
        default=3.0,
        help="普通趋势 RR（默认 3.0）",
    )
    p.add_argument(
        "--sl-mult-strong",
        type=float,
        default=3.5,
        help="强趋势 ATR 止损倍数（默认 3.5）",
    )
    p.add_argument(
        "--sl-mult-normal",
        type=float,
        default=3.0,
        help="普通趋势 ATR 止损倍数（默认 3.0）",
    )
    p.add_argument(
        "--refresh-seconds",
        type=int,
        default=10,
        help="每次刷新行情与信号的间隔秒数，建议 >= 5",
    )
    p.add_argument(
        "--initial-equity",
        type=float,
        default=10_000.0,
        help="初始资金（用于模型账户，若实盘模式下能获取真实权益，可在启动时覆盖）",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="启用实盘模式（连接 Binance USDT-M Futures），仍需在 config_live_v31.yaml 中设置 enable_trading 才会真实下单",
    )
    return p.parse_args()


def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    binance_client = None
    if args.live:
        cfg_yaml = load_live_config("config_live_v31.yaml")
        binance_client = BinanceFuturesV31(cfg_yaml)

    cfg_live = LiveEngineConfig(
        symbols=symbols,
        timeframe="5m",
        topk=args.topk,
        leverage=args.leverage,
        risk_per_trade=args.risk_per_trade,
        refresh_seconds=args.refresh_seconds,
        initial_equity=args.initial_equity,
        live_mode=args.live,
        rr_strong=args.rr_strong,
        rr_normal=args.rr_normal,
        sl_mult_strong=args.sl_mult_strong,
        sl_mult_normal=args.sl_mult_normal,
    )

    engine = LiveTradeEngineV31_13(cfg_live, binance=binance_client)
    engine.run_forever()


if __name__ == "__main__":
    main()
