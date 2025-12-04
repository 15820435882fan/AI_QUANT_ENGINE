# -*- coding: utf-8 -*-
"""
live_trade_engine_v31_6.py

V31_7 · 多币轮动 · Binance USDT-M 永续合约 实盘引擎（实时循环版）

相对 V31_4 的关键升级：
- 行情从 Binance 拉取固定 500 根 5m K 线（不再只拿 1 根），指标可正常计算
- 评分不再恒为 0
- 保持与 V31_1 回测相同的指标与信号逻辑
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
        期望 YAML 结构：

        binance:
          name: binance
          apiKey: "..."
          secret: "..."
          enableRateLimit: true
          options:
            defaultType: future
            adjustForTimeDifference: true
          proxies:
            http: "http://127.0.0.1:26217"
            https: "http://127.0.0.1:26217"
          enable_trading: false
        """
        ex_cfg = cfg.get("binance") or cfg.get("exchange") or {}

        name = ex_cfg.get("name", "binance")
        api_key = ex_cfg.get("apiKey") or ex_cfg.get("api_key")
        secret = ex_cfg.get("secret") or ex_cfg.get("api_secret")
        if not api_key or not secret:
            raise ValueError("Binance 配置缺少 apiKey / secret")

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

    # ---- 行情 ----
    def fetch_ohlcv_5m(self, symbol: str, limit: int = 500) -> Optional[pd.DataFrame]:
        """
        固定拉取最近 limit 根 5m K 线，默认 500 根。
        不再用 since，避免只拿到 1 根 K 线的尴尬情况。
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

    # ---- 账户 ----
    def fetch_futures_equity_usdt(self) -> Optional[float]:
        """
        获取 USDT-M 永续账户 USDT 权益
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


    def place_sl_tp_orders(
        self,
        symbol: str,
        side: str,
        qty: float,
        sl_price: float,
        tp_price: float,
        position_side: Optional[str] = None,
    ):
        """
        为当前持仓挂一组：止损限价单 + 止盈市价单（reduceOnly）。
        side 为开仓方向（"buy"/"sell"），本方法会自动用反向 side 挂单。
        """
        if qty <= 0:
            print(f"[Binance] SL/TP 数量为 0，跳过: {symbol} qty={qty}")
            return None, None

        opposite = "sell" if side == "buy" else "buy"
        params_common = {
            "reduceOnly": True,
            "timeInForce": "GTC",
        }
        if position_side:
            params_common["positionSide"] = position_side

        sl_order = None
        tp_order = None

        # 止损：止损限价单
        try:
            sl_order = self.ex.create_order(
                symbol=symbol,
                type="STOP_LOSS_LIMIT",
                side=opposite,
                amount=qty,
                price=sl_price,
                params={
                    **params_common,
                    "stopPrice": sl_price,
                    "workingType": "CONTRACT_PRICE",
                },
            )
            print(f"[Binance] SL 挂单成功: {sl_order}")
        except Exception as e:
            print(f"[Binance] SL 下单失败: {symbol}, err={e}")

        # 止盈：止盈市价单
        try:
            tp_order = self.ex.create_order(
                symbol=symbol,
                type="TAKE_PROFIT_MARKET",
                side=opposite,
                amount=qty,
                params={
                    **params_common,
                    "stopPrice": tp_price,
                    "workingType": "CONTRACT_PRICE",
                },
            )
            print(f"[Binance] TP 挂单成功: {tp_order}")
        except Exception as e:
            print(f"[Binance] TP 下单失败: {symbol}, err={e}")

        return sl_order, tp_order

    def fetch_positions(self):
        """获取当前所有持仓信息"""
        try:
            return self.ex.fetch_positions()
        except Exception as e:
            print(f"[Binance] fetch_positions 失败: {e}")
            return []

    def current_position_info(self) -> str:
        """简单汇总当前持仓信息，用于状态打印"""
        try:
            positions = self.fetch_positions()
            active = []
            for p in positions:
                contracts = float(p.get("contracts") or 0)
                if contracts <= 0:
                    continue
                symbol = p.get("symbol", "")
                side = p.get("side", "")
                upnl = float(p.get("unrealizedPnl") or 0.0)
                active.append(f"{symbol} {side} qty={contracts} upnl={upnl:.2f}")
            return "; ".join(active) if active else "无持仓"
        except Exception as e:
            return f"查询失败: {e}"

    def calc_unrealized_pnl_usdt(self) -> float:
        """汇总所有持仓的未实现 PnL（USDT）"""
        total = 0.0
        try:
            positions = self.fetch_positions()
            for p in positions:
                upnl = float(p.get("unrealizedPnl") or 0.0)
                total += upnl
        except Exception as e:
            print(f"[Binance] 计算未实现 PnL 失败: {e}")
        return total


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
    live_mode: bool = False


# ==============================
# 多币轮动实盘引擎（实时循环版）
# ==============================
class LiveTradeEngineV31_6:
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

        # 抽水 + 复利账户
        self.account = WaterfallAccountV31_1(
            initial_capital=cfg_live.initial_equity,
            withdraw_rate=0.10,
            growth_threshold=0.01,
            enable_waterfall=True,
        )

        # 尝试接入实盘权益
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

        # 记录最近一次同步的实盘权益，用于计算真实 PnL → 驱动抽水 + 复利
        self.last_equity_real: float = float(self.account.total_equity())

        self.consecutive_losses: Dict[str, int] = {s: 0 for s in cfg_live.symbols}

        # 统计相关
        self.start_ts = time.time()      # 程序启动时间
        self.loop_count = 0              # 刷新次数
        self.total_trades = 0            # 总开仓次数
        self.realized_pnl = 0.0          # 已实现收益（通过实盘权益差计算）
        self.unrealized_pnl = 0.0        # 未实现浮盈（从持仓中获取）

    # ---------- 趋势评分 ----------
    def _calc_trend_score(self, df5: pd.DataFrame) -> float:
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

    # ---------- 单轮计算 ----------
    def run_one_cycle(self):
        print(
            f"\n==== V31_7 周期刷新 · symbols={self.cfg_live.symbols}, "
            f"topk={self.cfg_live.topk}, live={self.cfg_live.live_mode}, "
            f"refresh={self.cfg_live.refresh_seconds}s ===="
        )

        if self.cfg_live.live_mode and self.binance is None:
            print("[Engine] live_mode=True 但未提供 Binance 客户端，直接返回。")
            return

        # 若在实盘模式下，先用 Binance USDT-M 实盘权益驱动账户账本（真实复利 + 抽水）
        if self.cfg_live.live_mode and self.binance is not None:
            eq_real = self.binance.fetch_futures_equity_usdt()
            if eq_real is not None and eq_real > 0:
                pnl_real = float(eq_real - self.last_equity_real)
                if abs(pnl_real) > 1e-8:
                    # 使用真实 PnL 推动 WaterfallAccount：既更新交易资金，又检查是否抽水
                    now_ts = pd.Timestamp.utcnow()
                    try:
                        self.account.apply_pnl(pnl_real, now_ts)
                    except Exception as e:
                        print(f"[Account] 应用实盘 PnL 时出错: {e}")
                    # 记录已实现收益，驱动统计与复利
                    self.realized_pnl += pnl_real
                    self.last_equity_real = float(eq_real)

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

            # 指标 NaN 用前值填一下
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
        else:
            sorted_syms = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
            topk_syms = [s for s, _ in sorted_syms[: self.cfg_live.topk]]
            scores_str = ", ".join([f"{k}: {v:.6f}" for k, v in score_map.items()])
            print(f"[TopK] 当前最强: {topk_syms}, scores={{ {scores_str} }}")

            # 3) TopK → 入场信号
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

                equity_for_risk = self.account.total_equity()
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

                qty = notional / entry_price
                direction_str = "多" if side == "LONG" else "空"
                print(
                    f"[Signal] {full_sym} @ {last_ts}, 开{direction_str}, "
                    f"notional={notional:.2f}, qty={qty:.6f}, entry≈{entry_price:.4f}, "
                    f"SL={stop_price:.4f}, TP={take_price:.4f}, RR={rr_used:.2f}"
                )

                # 真正下单 + 挂止盈/止损
                if self.cfg_live.live_mode and self.binance is not None:
                    side_str = "buy" if side == "LONG" else "sell"
                    amount = qty / self.cfg_live.leverage

                    order = self.binance.create_market_order(full_sym, side_str, amount)
                    if order is not None:
                        pos_side = "LONG" if side_str == "buy" else "SHORT"
                        try:
                            self.binance.place_sl_tp_orders(
                                symbol=full_sym,
                                side=side_str,
                                qty=amount,
                                sl_price=stop_price,
                                tp_price=take_price,
                                position_side=pos_side,
                            )
                        except Exception as e:
                            print(f"[Engine] 挂 SL/TP 时出错: {e}")
                        self.total_trades += 1

        # 4) 账户打印 + 统计信息
        equity = self.account.total_equity()
        trading = self.account.risk_capital()
        profit_pool = self.account.profit_pool
        print(
            f"[Account] 模型层资金: equity={equity:.2f}, trading={trading:.2f}, "
            f"profit_pool={profit_pool:.2f}"
        )

        eq_real = None
        if self.cfg_live.live_mode and self.binance is not None:
            eq_real = self.binance.fetch_futures_equity_usdt()
            if eq_real is not None:
                print(f"[Account] Binance 实盘 USDT 权益: {eq_real:.2f}")
            # 更新未实现浮盈
            self.unrealized_pnl = self.binance.calc_unrealized_pnl_usdt()

        # 刷新统计
        self.loop_count += 1
        runtime_sec = time.time() - self.start_ts
        runtime_min = runtime_sec / 60.0
        eq_real_str = f"{eq_real:.2f}" if eq_real is not None else "N/A"

        pos_info = "N/A"
        if self.cfg_live.live_mode and self.binance is not None:
            pos_info = self.binance.current_position_info()

        print(
            "\n==== [Engine Status · V31_7] ============================\n"
            f"启动时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_ts))}\n"
            f"运行时长: {runtime_min:.2f} 分钟\n"
            f"刷新次数: {self.loop_count}\n"
            f"总开仓数: {self.total_trades}\n\n"
            f"模型总权益: {equity:.2f}\n"
            f"  - 交易资金: {trading:.2f}\n"
            f"  - 利润池:   {profit_pool:.2f}\n"
            f"实盘 USDT 权益: {eq_real_str}\n\n"
            f"已实现收益: {self.realized_pnl:.2f}\n"
            f"未实现浮盈: {self.unrealized_pnl:.2f}\n"
            f"当前持仓: {pos_info}\n"
            "=========================================================\n"
        )

    # ---------- 主循环 ----------
    def run_forever(self):
        print("==== 启动 V31_7 多币轮动·Binance USDT-M Futures 实盘引擎（实时循环） ====")
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
    p = argparse.ArgumentParser(description="V31_7 多币轮动 · Binance USDT-M Futures 实时引擎")
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
    )

    engine = LiveTradeEngineV31_6(cfg_live, binance=binance_client)
    engine.run_forever()


if __name__ == "__main__":
    main()

