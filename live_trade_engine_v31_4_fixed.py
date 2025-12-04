# -*- coding: utf-8 -*-
"""
live_trade_engine_v31_4.py

V31_4 · 多币轮动 · Binance USDT-M 永续合约 实盘引擎（实时循环版）

功能概览：
- 使用 v31_core_v31_1 提供的核心策略内核：
  - V31CoreConfig
  - WaterfallAccountV31_1
  - build_multi_tf_indicators
  - entry_signal_v31
  - compute_sl_tp_notional_v31
- 多币种 TopK 轮动（5m 主周期），实时从 Binance Futures 获取行情
- 每 N 秒刷新一次（默认 10s），持续运行，直到手动 Ctrl+C 终止
- 可选实盘模式：
  - --live 打开与 Binance 的连接
  - config_live_v31.yaml 中 binance.enable_trading=True 才真正下单

当前版本说明：
- ✅ 实时行情 + TopK + 策略信号
- ✅ 读取 Binance USDT-M 实盘权益并打印
- ✅ 仿真账户使用 WaterfallAccountV31_1（抽水 + 复利）
- ✅ 可在有信号时按策略仓位模型计算下单数量，并调用 Binance 市价单
- ⚠ 未实现持仓生命周期管理（自动平仓 / 止盈止损），当前仅示范开仓逻辑
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
    """
    加载实盘配置文件（主要用于 Binance API 参数和一些全局设置）。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到实盘配置文件: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


# ==============================
# Binance USDT-M Futures 适配器
# ==============================
class BinanceFuturesV31:
    """
    简单封装 ccxt.binance，用于 USDT-M 永续合约：
    - fetch_ohlcv_5m: 获取 5m K 线
    - fetch_futures_equity_usdt: 获取 USDT 总权益（尽力解析）
    - create_market_order: 市价单开仓（受 enable_trading 控制）
    """

    def __init__(self, cfg: dict):
        """
        cfg 期望包含形如：

        binance:
          name: binance
          apiKey: "xxx"
          secret: "yyy"
          enableRateLimit: true
          options:
            defaultType: future
            adjustForTimeDifference: true
          enable_trading: false
        """
        ex_cfg = cfg.get("binance") or cfg.get("exchange") or {}

        name = ex_cfg.get("name", "binance")
        api_key = ex_cfg.get("apiKey") or ex_cfg.get("api_key")
        secret = ex_cfg.get("secret") or ex_cfg.get("api_secret")

        if not api_key or not secret:
            raise ValueError("Binance 配置缺少 apiKey / secret")

        # 是否真正允许下单的开关（双重保护：命令行 --live + 配置 enable_trading）
        self.enable_trading: bool = bool(ex_cfg.get("enable_trading", False))

        # 剩余参数交给 ccxt，例如 enableRateLimit / options 等
        extra = {
            k: v
            for k, v in ex_cfg.items()
            if k not in ["name", "apiKey", "api_key", "secret", "api_secret", "enable_trading"]
        }

        ex_class = getattr(ccxt, name)
        self.ex = ex_class({
            "apiKey": api_key,
            "secret": secret,
            **extra,
        })

    # ---- 行情 ----
    def fetch_ohlcv_5m(self, symbol: str, lookback_minutes: int = 60 * 24) -> Optional[pd.DataFrame]:
        """
        获取指定 symbol 的近一段时间 5m K 线。
        lookback_minutes: 回溯分钟数，例如 60*24 = 1 天。
        """
        now_ms = int(self.ex.milliseconds())
        since_ms = now_ms - lookback_minutes * 60 * 1000
        try:
            ohlcv = self.ex.fetch_ohlcv(symbol, timeframe="5m", since=since_ms)
        except Exception as e:
            print(f"[Binance] fetch_ohlcv 失败: symbol={symbol}, err={e}")
            return None

        if not ohlcv:
            return None

        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        return df

    # ---- 账户 ----
    def fetch_futures_equity_usdt(self) -> Optional[float]:
        """
        尝试从 USDT-M futures 账户中获取 USDT 总权益。
        注意：不同 ccxt 版本 / 账户模式下结构略有不同，这里做一些兼容处理。
        """
        try:
            # 对于部分 ccxt 版本，可以传 type 指定 futures
            balance = self.ex.fetch_balance(params={"type": "future"})
        except Exception as e:
            print(f"[Binance] 获取期货账户余额失败: {e}")
            try:
                balance = self.ex.fetch_balance()
            except Exception as e2:
                print(f"[Binance] 退回普通 fetch_balance 也失败: {e2}")
                return None

        eq = None

        # 常见结构 1：balance['total']['USDT']
        total = balance.get("total")
        if isinstance(total, dict):
            eq = total.get("USDT")

        # 常见结构 2：balance['USDT']['total'] 或 ['free']
        if eq is None and "USDT" in balance:
            usdt_info = balance["USDT"]
            if isinstance(usdt_info, dict):
                eq = usdt_info.get("total") or usdt_info.get("free")

        # 如仍解析失败，打印一次结构以便调试
        if eq is None:
            print("[Binance] 警告：未能从 fetch_balance 中解析出 USDT 总权益，原始结构如下（仅打印一次）：")
            print(balance)

        return float(eq) if eq is not None else None

    # ---- 下单 ----
    def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        params: Optional[dict] = None,
    ):
        """
        市价单下单接口：
        - side: 'buy' 或 'sell'
        - amount: 合约张数（或币的数量，具体取决于合约规格）
        """
        params = params or {}

        if amount <= 0:
            print(f"[Binance] 下单数量为 0，跳过: {symbol} {side} {amount}")
            return None

        if not self.enable_trading:
            print(f"[Binance] enable_trading=False，跳过真实下单: {symbol} {side} {amount}")
            return None

        print(f"[Binance] 准备下单: {symbol} {side} {amount}")
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
    leverage: float = 10.0
    risk_per_trade: float = 0.01
    refresh_seconds: int = 10
    initial_equity: float = 10_000.0
    live_mode: bool = False          # 是否连接交易所
    lookback_minutes: int = 60 * 24  # 每轮从 Binance 拉取多少分钟的历史数据用于计算指标


# ==============================
# 多币轮动实盘引擎（实时循环版）
# ==============================
class LiveTradeEngineV31_4:
    def __init__(
        self,
        cfg_live: LiveEngineConfig,
        binance: Optional[BinanceFuturesV31] = None,
    ):
        self.cfg_live = cfg_live
        self.binance = binance

        # 策略核心配置（与 V31_1 一致）
        self.core_cfg = V31CoreConfig(
            symbol="MULTI",
            days=365,
            leverage=cfg_live.leverage,
            risk_per_trade=cfg_live.risk_per_trade,
        )

        # 账户模型：抽水 + 复利（模型层资金）
        self.account = WaterfallAccountV31_1(
            initial_capital=cfg_live.initial_equity,
            withdraw_rate=0.10,
            growth_threshold=0.01,
            enable_waterfall=True,
        )

        # 如果是实盘模式，尝试用 Binance 实盘 USDT 权益覆盖模型初始资金
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

        # 连续亏损计数（后续若实现持仓管理，可用此字段）
        self.consecutive_losses: Dict[str, int] = {s: 0 for s in cfg_live.symbols}

    # ---------- 趋势评分（用于 TopK 排序） ----------
    def _calc_trend_score(self, df5: pd.DataFrame) -> float:
        """
        简单趋势评分：使用近期收益率 + 趋势强度 + 趋势方向。
        """
        if df5 is None or df5.empty or len(df5) < 60:
            return 0.0

        # 最近约 24 小时
        recent = df5.iloc[-12 * 24 :]
        if recent.empty:
            return 0.0

        close = recent["close"]
        ret = float(close.iloc[-1] / close.iloc[0] - 1.0)

        strength = float(recent["trend_strength"].mean())
        dir_mean = float(recent["trend_dir"].mean())

        score = ret * 100.0 + strength * 5.0 + dir_mean * 2.0
        return score

    # ---------- 单轮计算（不会 sleep） ----------
    def run_one_cycle(self):
        print(
            f"\n==== V31_4 周期刷新 · symbols={self.cfg_live.symbols}, "
            f"topk={self.cfg_live.topk}, live={self.cfg_live.live_mode}, "
            f"refresh={self.cfg_live.refresh_seconds}s ===="
        )

        if self.cfg_live.live_mode and self.binance is None:
            print("[Engine] live_mode=True 但未提供 Binance 客户端，直接返回。")
            return

        df_map: Dict[str, pd.DataFrame] = {}
        score_map: Dict[str, float] = {}

        # 1) 拉取多币种 5m K 线 & 指标
        for sym in self.cfg_live.symbols:
            full_sym = sym.upper()
            if not full_sym.endswith("USDT"):
                full_sym += "USDT"

            df_raw = None
            if self.binance is not None:
                df_raw = self.binance.fetch_ohlcv_5m(
                    full_sym,
                    lookback_minutes=self.cfg_live.lookback_minutes,
                )

            if df_raw is None or df_raw.empty:
                print(f"[Engine] 无法获取 {full_sym} 的实时 K 线，跳过。")
                continue

            # 指标计算（1H/4H HMA+ADX, ATR, BOLL, EMA, MACD 等）
            ctx = build_multi_tf_indicators(df_raw, self.core_cfg)
            df5 = ctx["df_5m"]
            df_map[full_sym] = df5
            score_map[full_sym] = self._calc_trend_score(df5)

        if not score_map:
            print("[Engine] 本轮未能获得任何币种的评分，结束本轮。")
            return

        # 2) TopK 选择
        sorted_syms = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        topk_syms = [s for s, _ in sorted_syms[: self.cfg_live.topk]]

        scores_str = ", ".join([f"{k}: {v:.6f}" for k, v in score_map.items()])
        print(f"[TopK] 当前最强: {topk_syms}, scores={{ {scores_str} }}")

        # 3) 对 TopK 中每个币种，根据最后一根 5m K 线给出一次入场信号
        for full_sym in topk_syms:
            df5 = df_map.get(full_sym)
            if df5 is None or len(df5) < 5:
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

            if notional <= 0 or stop_price <= 0 or take_price <= 0:
                continue

            qty = notional / entry_price  # 名义仓位对应的「币数量」（未除杠杆）
            direction_str = "多" if side == "LONG" else "空"
            print(
                f"[Signal] {full_sym} @ {last_ts}, 开{direction_str}, "
                f"notional={notional:.2f}, qty={qty:.6f}, entry≈{entry_price:.4f}, "
                f"SL={stop_price:.4f}, TP={take_price:.4f}, RR={rr_used:.2f}"
            )

            # 实盘下单（注意：当前版本尚未实现自动平仓逻辑，请谨慎开启 enable_trading）
            if self.cfg_live.live_mode and self.binance is not None:
                side_str = "buy" if side == "LONG" else "sell"
                # 合约场景下，这里采用「按杠杆折算后的数量」，你可以根据实际合约规格调整
                amount = qty / self.cfg_live.leverage
                self.binance.create_market_order(full_sym, side_str, amount)

        # 4) 打印账户当前状态（模型层）
        equity = self.account.total_equity()
        trading = self.account.risk_capital()
        profit_pool = self.account.profit_pool
        print(
            f"[Account] 模型层资金: equity={equity:.2f}, trading={trading:.2f}, "
            f"profit_pool={profit_pool:.2f}"
        )

        # 如有 Binance，则展示实盘 USDT 权益
        if self.cfg_live.live_mode and self.binance is not None:
            eq_real = self.binance.fetch_futures_equity_usdt()
            if eq_real is not None:
                print(f"[Account] Binance 实盘 USDT 权益: {eq_real:.2f}")

    # ---------- 主循环 ----------
    def run_forever(self):
        print("==== 启动 V31_4 多币轮动·Binance USDT-M Futures 实盘引擎（实时循环） ====")
        print(f"symbols = {self.cfg_live.symbols}, timeframe={self.cfg_live.timeframe}, topk={self.cfg_live.topk}")
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
    p = argparse.ArgumentParser(description="V31_4 多币轮动 · Binance USDT-M Futures 实时引擎")
    p.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT",
        help="监控币种列表，用逗号分隔，例如: BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT",
    )
    p.add_argument("--topk", type=int, default=2, help="每轮选择最强 TopK 个币种")
    p.add_argument("--leverage", type=float, default=10.0, help="杠杆倍数")
    p.add_argument(
        "--risk-per-trade",
        type=float,
        default=0.01,
        help="单笔风险占比，例如 0.01 = 1%",
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
        help="初始资金（用于模型账户，若实盘模式下能获取真实权益，则会被覆盖）",
    )
    p.add_argument(
        "--lookback-minutes",
        type=int,
        default=60 * 24,
        help="每轮从 Binance 拉取多少分钟的 5m K 线用于计算指标，例如 1440 = 1 天",
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
        lookback_minutes=args.lookback_minutes,
    )

    engine = LiveTradeEngineV31_4(cfg_live, binance=binance_client)
    engine.run_forever()


if __name__ == "__main__":
    main()
