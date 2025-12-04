# -*- coding: utf-8 -*-
"""
live_trade_engine_v31_15.py

V31_15 · AutoDetect_Failsafe · 多币轮动 · Binance USDT-M 永续合约 实盘引擎

在 V31_14_Failsafe 的基础上，做了以下增强：

1）继承 V31_13 / V31_14 的全部交易逻辑
   - 信号、开仓、平仓、风控、TopK、多币轮动、Waterfall 资金管理等完全不变
   - 仍然依赖 v31_core_v31_1 里的核心函数和配置，确保与 V31_1 回测版逻辑严格一致

2）保持 Failsafe 机制：
   - 平仓下单失败时，自动重新同步该币的交易所仓位
   - 若仍有仓位 → 用降级模式再次强制平仓
   - 若已无仓位 → 本地删除该仓位，避免“程序以为有仓，交易所已经没仓”的错位问题

3）新增 AutoDetect 能力（关键修复点）：
   - 在 Failsafe 同步持仓时，针对 positionSide = "BOTH" 的场景：
       * 若 positionAmt > 0 → 视为 LONG 仓位
       * 若 positionAmt < 0 → 视为 SHORT 仓位
   - 这样可以兼容：
       * 从单向模式切换到对冲模式的历史仓位
       * ccxt / 交易所返回的 net 模式/BOTH 结构
   - 不再漏掉这类仓位，Failsafe 能正常识别并强制平仓

4）删除上一版 V31_15 误追加的“第二套简化引擎”和本地 compute_sl_tp_notional_v31 重定义：
   - 彻底避免覆盖 v31_core_v31_1 中的官方风险模型
   - 让实盘引擎与回测核心保持 1:1 一致性
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional
import time
import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
import builtins
from notify import TelegramNotifier
import yaml
import ccxt
import pandas as pd

from collections import defaultdict

# ==============================
# 日志与速率限制工具
# ==============================
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("live_trade_engine_v31")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    log_path = os.path.join(LOG_DIR, "live_trade_engine_v31.log")
    _file_handler = TimedRotatingFileHandler(
        log_path, when="midnight", backupCount=7, encoding="utf-8"
    )
    _file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_file_handler)

# 同步所有 print 到日志
_orig_print = builtins.print
def patched_print(*args, **kwargs):
    _orig_print(*args, **kwargs)
    try:
        msg = " ".join(str(a) for a in args)
        logger.info(msg)
    except Exception:
        pass

builtins.print = patched_print
print = patched_print


class RateLimiter:
    """简单的速率限制器，避免触发交易所频控"""
    def __init__(self, min_interval: float = 0.12):
        self.min_interval = float(min_interval)
        self._last_ts = 0.0

    def wait(self):
        now = time.time()
        dt = now - self._last_ts
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last_ts = time.time()

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
        cfg: 一般来自 config_live_v31.yaml
        负责：
        - 初始化 ccxt 交易所实例
        - 检测 / 决定 hedge_mode（对冲模式 vs 单向持仓）
        - 提供 fetch_ohlcv_5m / fetch_futures_equity_usdt / create_market_order 等接口
        """
        ex_cfg = cfg.get("binance") or cfg.get("exchange") or {}

        name = ex_cfg.get("name", "binance")
        api_key = ex_cfg.get("apiKey") or ex_cfg.get("api_key")
        secret = ex_cfg.get("secret") or ex_cfg.get("api_secret")
        if not api_key or not secret:
            raise ValueError("Binance 配置缺少 apiKey / secret")

        self.enable_trading: bool = bool(ex_cfg.get("enable_trading", False))
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
        # 速率限制与重试参数
        self.rate_limiter = RateLimiter(min_interval=ex_cfg.get("min_interval", 0.12))
        self.max_retries = int(ex_cfg.get("max_retries", 3))
        self.retry_delay = float(ex_cfg.get("retry_delay", 1))

        # 1）若配置文件中显式指定 hedge_mode，则以配置为准
        explicit_hedge = ex_cfg.get("hedge_mode")
        if explicit_hedge is not None:
            self.hedge_mode = bool(explicit_hedge)
            mode_str = "对冲模式(双向持仓)" if self.hedge_mode else "单向持仓模式"
            print(f"[Binance] 配置文件中显式指定 hedge_mode={self.hedge_mode} → {mode_str}")
            return

        # 2）否则尝试根据 positionSide 自动检测
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
                self.hedge_mode = False
                print(
                    "[Binance] 无 positionSide 字段信息，默认按单向持仓模式处理 "
                    f"(sides={sides})"
                )
            else:
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
            self.hedge_mode = False
            print(f"[Binance] 警告：无法自动检测持仓模式，默认按单向持仓模式处理。err={e}")


    def _call_with_retry(self, func, *args, **kwargs):
        """统一带重试与速率限制的调用封装"""
        retries = getattr(self, "max_retries", 3)
        delay = getattr(self, "retry_delay", 1.0)
        last_err = None
        for attempt in range(retries):
            try:
                if hasattr(self, "rate_limiter") and self.rate_limiter is not None:
                    self.rate_limiter.wait()
                return func(*args, **kwargs)
            except Exception as e:
                last_err = e
                print(f"[Binance] 调用 {getattr(func, '__name__', func)} 失败，第 {attempt+1} 次重试: {e}")
                time.sleep(delay)
        print(f"[Binance] 多次重试后仍失败: {last_err}")
        raise last_err

    # ---- 行情 ----

    def fetch_ohlcv_5m(self, symbol: str, limit: int = 500) -> Optional[pd.DataFrame]:
        try:
            ohlcv = self._call_with_retry(self.ex.fetch_ohlcv, symbol, timeframe="5m", limit=limit)
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
        优先用期货账户的 USDT total，退而求其次用普通账户余额里的 USDT。
        """
        try:
            balance = self._call_with_retry(self.ex.fetch_balance, params={"type": "future"})
        except Exception as e:
            print(f"[Binance] 获取期货账户余额失败: {e}")
            try:
                balance = self._call_with_retry(self.ex.fetch_balance)
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
            print("[Binance] 警告：未能解析出 USDT 总权益")
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
        params: 可能包含
            - reduceOnly
            - positionSide（对冲模式时）
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
            order = self._call_with_retry(
                self.ex.create_order,
                symbol=symbol,
                type="market",
                side=side,
                amount=amount,
                params=params,
            )
            # 精简成功日志，避免输出整份订单字典
            avg_price = None
            try:
                info = order.get("info") or {}
                avg_price = info.get("avgPrice") or info.get("price")
            except Exception:
                avg_price = None
            if avg_price:
                print(f"[Binance] 下单成功: {symbol} {side} {amount} @≈{avg_price}")
            else:
                print(f"[Binance] 下单成功: {symbol} {side} {amount}")
            return order
        except Exception as e:
            print(f"[Binance] 下单失败: {e}")
            return None


# ==============================
# 实盘引擎配置
# ==============================
@dataclass
class LiveEngineConfig:
    symbols: List[str]
    timeframe: str = "5m"
    topk: int = 2
    leverage: float = 3.0
    risk_per_trade: float = 0.01
    refresh_seconds: int = 10
    initial_equity: float = 10_000.0
    live_mode: bool = False
    rr_strong: float = 4.0
    rr_normal: float = 3.0
    sl_mult_strong: float = 3.5
    sl_mult_normal: float = 3.0


# ==============================
# 实盘持仓结构
# ==============================
@dataclass
class LivePositionV31:
    symbol: str
    side: str   # "LONG" or "SHORT"
    entry_ts: pd.Timestamp
    entry_price: float
    qty: float
    notional: float
    stop_price: float
    take_price: float
    max_bars: int
    bars_held: int = 0
    last_bar_ts: Optional[pd.Timestamp] = None


# ==============================
# 多币轮动实盘引擎 · V31_15 AutoDetect_Failsafe
# ==============================
class LiveTradeEngineV31_15:
    def __init__(
        self,
        cfg_live: LiveEngineConfig,
        binance: Optional[BinanceFuturesV31] = None,
    ):
        self.cfg_live = cfg_live
        self.binance = binance

        # === V31 核心配置 ===
        self.core_cfg = V31CoreConfig(
            symbol="MULTI",
            days=365,
            leverage=cfg_live.leverage,
            risk_per_trade=cfg_live.risk_per_trade,
        )
        self.core_cfg.rr_strong = cfg_live.rr_strong
        self.core_cfg.rr_normal = cfg_live.rr_normal
        self.core_cfg.sl_mult_strong = cfg_live.sl_mult_strong
        self.core_cfg.sl_mult_normal = cfg_live.sl_mult_normal

        # === Waterfall 资金管理 ===
        self.account = WaterfallAccountV31_1(
            initial_capital=cfg_live.initial_equity,
            withdraw_rate=0.0,
            growth_threshold=0.01,
            enable_waterfall=False,
        )

        # 若是 live 模式，使用实盘权益覆盖 initial_equity
        if self.cfg_live.live_mode and self.binance is not None:
            eq = self.binance.fetch_futures_equity_usdt()
            if eq is not None and eq > 0:
                self.account.initial_capital = float(eq)
                self.account.trading_capital = float(eq)
                self.account.last_high = float(eq)
                print(f"[Account] 使用 Binance 实盘 USDT 权益作为初始资金: {eq:.2f}")
            else:
                print(
                    f"[Account] 未能获取实盘权益，继续使用 initial_equity={cfg_live.initial_equity:.2f}"
                )

        # 连续亏损计数（按币种）
        self.consecutive_losses: Dict[str, int] = {s: 0 for s in cfg_live.symbols}
        # 本地持仓记录
        self.positions: Dict[str, LivePositionV31] = {}

        # 是否已做过一次交易所持仓同步
        self.synced_positions_from_exchange: bool = False

        self.start_time = pd.Timestamp.utcnow()
        self.refresh_count: int = 0
        self.total_signal_count: int = 0
        self.total_order_sent: int = 0
        self.total_order_success: int = 0
        self.total_order_fail: int = 0
        # 最近一次下单失败原因（只保留一条，便于排查）
        self.last_fail_reason: str = ""
        # 按币种统计 PnL（win: 盈利累积, loss: 亏损累积, sum: 净值）
        self.pnl_stats = defaultdict(lambda: {"win": 0.0, "loss": 0.0, "sum": 0.0})

    # ---------- 趋势评分（与 V31_14 相同） ----------
    def _calc_trend_score(self, df5: pd.DataFrame) -> float:
        if df5 is None or df5.empty or len(df5) < 60:
            return 0.0

        recent = df5.iloc[-12 * 24 :]
        if recent.empty:
            return 0.0

        close = recent["close"]
        ret = float(close.iloc[-1] / close.iloc[0] - 1.0)
        strength = float(recent["trend_strength"].mean())
        dir_mean = float(recent["trend_dir"].mean())

        score = ret * 100.0 + strength * 5.0 + dir_mean * 2.0
        return score

    # ---------- 启动后同步账户已有仓位 ----------
    def _normalize_symbol(self, sym: str) -> str:
     """
     将各种格式的 symbol 统一成诸如 BTCUSDT / ETHUSDT 这种形式，便于匹配：
     - "ETHUSDT"        -> "ETHUSDT"
     - "ETH/USDT"       -> "ETHUSDT"
     - "ETH/USDT:USDT"  -> "ETHUSDT"
     """
     if not sym:
         return ""
     s = str(sym).upper()
     # 去掉 futures 类型后缀，例如 ETH/USDT:USDT -> ETH/USDT
     if ":" in s:
         s = s.split(":", 1)[0]
     # 把 ETH/USDT -> ETHUSDT
     if "/" in s:
         parts = s.split("/")
         if len(parts) == 2:
             s = parts[0] + parts[1]
     return s

    def _sync_positions_from_exchange(self, df_map: Dict[str, pd.DataFrame]):
        """
        V31_16 · AutoSync 版本：
        - 启动后自动接管交易所已有仓位
        - 同时兼容：
            * balance["info"]["positions"]（Binance 原始返回）
            * balance["positions"]（部分 ccxt 版本）
            * fetch_positions()（ccxt 标准化格式，symbol 形如 "ETH/USDT:USDT"）
        - 统一做 symbol 归一化 + side/qty/entryPrice 抽取
        """
        if self.synced_positions_from_exchange:
            return
        if not self.cfg_live.live_mode or self.binance is None:
            return False, "非实盘模式或未提供交易所客户端"

        # 收集所有可能来源的原始持仓记录
        raw_positions = []

        # --- 来源 1：fetch_balance(type="future").info.positions ---
        try:
            bal = self.binance._call_with_retry(self.binance.ex.fetch_balance, params={"type": "future"})
        except Exception as e:
            print(f"[Sync] 获取期货账户持仓失败(fetch_balance)，尝试其他方式: {e}")
            bal = None

        if isinstance(bal, dict):
            info = bal.get("info") or {}
            pos1 = info.get("positions") or []
            if isinstance(pos1, list) and pos1:
                raw_positions.extend(
                    {"_source": "balance.info.positions", **p} for p in pos1
                )

            # 有些 ccxt 版本会把标准化 positions 直接挂 balance["positions"]
            pos2 = bal.get("positions") or []
            if isinstance(pos2, list) and pos2:
                raw_positions.extend(
                    {"_source": "balance.positions", **p} for p in pos2
                )

        # --- 来源 2：fetch_positions()（若支持） ---
        try:
            if getattr(self.binance.ex, "has", {}).get("fetchPositions"):
                pos3 = self.binance._call_with_retry(self.binance.ex.fetch_positions)
                if isinstance(pos3, list) and pos3:
                    raw_positions.extend(
                        {"_source": "fetch_positions", **p} for p in pos3
                    )
        except Exception as e:
            print(f"[Sync] fetch_positions() 调用失败，忽略该来源: {e}")

        if not raw_positions:
            print("[Sync] 未从任何来源获取到持仓记录，放弃接管。")
            self.synced_positions_from_exchange = True
            return

        # 去重 & 汇总：按 (symbol_norm, side) 聚合
        symbols_set = {
            s.upper() if s.upper().endswith("USDT") else (s.upper() + "USDT")
            for s in self.cfg_live.symbols
        }
        agg = {}  # key: (symbol_norm, side) -> dict

        for p in raw_positions:
            sym_raw = p.get("symbol") or (p.get("info") or {}).get("symbol")
            sym_norm = self._normalize_symbol(sym_raw)
            if sym_norm not in symbols_set:
                continue

            # 数量：优先 positionAmt，其次 contracts
            amt = p.get("positionAmt", None)
            if amt is None:
                amt = p.get("contracts", None)
            try:
                amt = float(amt or 0.0)
            except Exception:
                continue
            if abs(amt) < 1e-8:
                continue  # 零仓位忽略

            # entryPrice / avgEntryPrice
            entry_price = (
                p.get("entryPrice")
                or p.get("avgEntryPrice")
                or p.get("entry")
                or 0.0
            )
            try:
                entry_price = float(entry_price or 0.0)
            except Exception:
                entry_price = 0.0

            # 对冲模式下优先 positionSide，单向模式下按正负判断
            side = None
            ps_raw = str(p.get("positionSide") or "").upper()
            side_ccxt = str(p.get("side") or "").upper()

            if ps_raw in ["LONG", "SHORT"]:
                side = ps_raw
            elif side_ccxt in ["LONG", "SHORT"]:
                side = side_ccxt
            else:
                # BOTH 或缺失，则用仓位数量正负判断
                side = "LONG" if amt > 0 else "SHORT"

            key = (sym_norm, side)
            notional = abs(amt) * entry_price if entry_price > 0 else 0.0

            agg[key] = {
                "symbol": sym_norm,
                "side": side,
                "qty": abs(amt),
                "entry_price": entry_price,
                "notional": notional,
                "_source": p.get("_source", "unknown"),
            }

        if not agg:
            print("[Sync] 虽然拿到了原始持仓数据，但都不符合过滤条件（symbol / qty / entry_price）。")
            self.synced_positions_from_exchange = True
            return

        synced_count = 0
        equity_for_risk = self.account.risk_capital()

        for (sym_norm, side), info_pos in agg.items():
            df5 = df_map.get(sym_norm)
            if df5 is None or len(df5) < 5:
                print(f"[Sync] {sym_norm} 虽有持仓，但本地 K 线不足，暂不接管。")
                continue

            last_ts = df5.index[-1]
            row = df5.iloc[-1]

            atr_1h = float(row.get("atr_1h", 0.0))
            if atr_1h <= 0:
                print(f"[Sync] {sym_norm} 无有效 atr_1h，暂不接管。")
                continue

            trend_strength = int(row.get("trend_strength", 0))

            entry_price = info_pos["entry_price"]
            if entry_price <= 0:
                # 若交易所没给出 entryPrice，则退化用当前 close 近似
                entry_price = float(row["close"])
                info_pos["entry_price"] = entry_price
                info_pos["notional"] = info_pos["qty"] * entry_price

            try:
                stop_price, take_price, _, max_bars, _, rr_used = compute_sl_tp_notional_v31(
                    cfg=self.core_cfg,
                    side=side,
                    trend_strength=trend_strength,
                    entry_price=entry_price,
                    atr_1h=atr_1h,
                    equity_for_risk=equity_for_risk,
                )
            except Exception as e:
                print(f"[Sync] 计算 {sym_norm} 的 SL/TP 失败，暂不接管: {e}")
                continue

            if stop_price <= 0 or take_price <= 0 or max_bars <= 0:
                print(f"[Sync] {sym_norm} 计算出的 SL/TP 非法，暂不接管。")
                continue

            lp = LivePositionV31(
                symbol=sym_norm,
                side=side,
                entry_ts=last_ts,
                entry_price=entry_price,
                qty=info_pos["qty"],
                notional=info_pos["notional"],
                stop_price=stop_price,
                take_price=take_price,
                max_bars=int(max_bars),
                bars_held=0,
                last_bar_ts=None,
            )

            self.positions[sym_norm] = lp
            synced_count += 1
            print(
                f"[Sync] 接管已有实盘持仓: {sym_norm}, side={side}, qty={info_pos['qty']}, "
                f"entry={entry_price:.4f}, SL={stop_price:.4f}, TP={take_price:.4f}, "
                f"max_bars={int(max_bars)}, notional={info_pos['notional']:.2f}, "
                f"RR_used={rr_used:.2f}, source={info_pos['_source']}"
            )

        if synced_count == 0:
            print("[Sync] 未发现需要接管的实盘持仓。")
        else:
            print(f"[Sync] 本次共接管 {synced_count} 个实盘持仓。")

        self.synced_positions_from_exchange = True


    # ---------- Failsafe：平仓失败后强制同步并二次处理 ----------
    def _failsafe_force_close(self, sym: str, pos: LivePositionV31):
        """
        场景：第一次平仓单被交易所拒绝（比如 reduceOnly 错误）。
        策略：
        1）重新从交易所获取该 symbol 的真实仓位情况
        2）若真实仓位已不存在 → 认为交易所已平仓，删除本地仓位
        3）若真实仓位仍存在 → 用“降级模式”再发一次平仓单：
            - hedge_mode=True  → side+positionSide 保持与仓位方向相反
            - reduceOnly=False → 避免 -1106

        V31_15 AutoDetect 增强：
        - 当 positionSide = "BOTH" 时，按 positionAmt 正负判断 LONG/SHORT：
            * positionAmt > 0 → LONG
            * positionAmt < 0 → SHORT
        """
        if not self.cfg_live.live_mode or self.binance is None:
            return

        try:
            bal = self.binance._call_with_retry(self.binance.ex.fetch_balance, params={"type": "future"})
        except Exception as e:
            print(f"[Failsafe] 获取期货账户余额失败，无法执行强制同步: {e}")
            return False, f"获取期货账户余额失败: {e}"

        info = bal.get("info") or {}
        positions = info.get("positions") or []
        sym_upper = sym.upper()

        real_side: Optional[str] = None   # "LONG" / "SHORT"
        real_qty: float = 0.0

        for p in positions:
            if str(p.get("symbol") or "").upper() != sym_upper:
                continue

            pos_amt = float(p.get("positionAmt") or 0.0)
            if abs(pos_amt) < 1e-8:
                continue

            ps_raw = str(p.get("positionSide") or "").upper()

            if self.binance.hedge_mode:
                # 对冲模式下，优先识别 LONG/SHORT，其次兼容 BOTH
                if ps_raw in ["LONG", "SHORT"]:
                    # 只关心与本地 pos.side 一致的记录
                    if ps_raw != pos.side:
                        continue
                    real_side = ps_raw
                    real_qty = abs(pos_amt)
                    break
                else:
                    # positionSide = "BOTH" 或空 → 用正负判断方向，并与 pos.side 对齐
                    inferred_side = "LONG" if pos_amt > 0 else "SHORT"
                    if inferred_side != pos.side:
                        continue
                    real_side = inferred_side
                    real_qty = abs(pos_amt)
                    break
            else:
                # 单向模式：只有一个 net 仓位，用正负判断方向
                inferred_side = "LONG" if pos_amt > 0 else "SHORT"
                real_side = inferred_side
                real_qty = abs(pos_amt)
                break

        if real_side is None or real_qty <= 0:
            # 说明交易所已经没有该方向仓位了，本地可以直接删除
            print(f"[Failsafe] {sym} 交易所已无对应仓位，认为已被其他方式平掉，删除本地记录。")
            self.positions.pop(sym, None)
            return True, ""

        # 实际仍然有仓位 → 再次尝试强制平仓（降级模式：不再使用 reduceOnly）
        side_str = "sell" if real_side == "LONG" else "buy"
        close_params: Dict[str, object] = {}
        if self.binance.hedge_mode:
            close_params["positionSide"] = real_side

        amount = real_qty
        print(
            f"[Failsafe] {sym} 仍有实盘仓位 side={real_side}, qty={real_qty}, "
            f"尝试以降级模式强制平仓: side={side_str}, amount={amount}"
        )

        try:
            order = self.binance.create_market_order(
                symbol=sym,
                side=side_str,
                amount=amount,
                params=close_params,
            )
        except Exception as e:
            print(f"[Failsafe] {sym} 二次平仓下单异常: {e}")
            order = None
            return False, f"二次平仓下单异常: {e}"

        if order is not None:
            print(f"[Failsafe] {sym} 二次平仓下单成功，本地删除仓位记录。")
            self.positions.pop(sym, None)
            return True, ""
        else:
            print(f"[Failsafe] {sym} 二次平仓仍失败，请人工检查交易所仓位。")
            return False, "二次平仓仍失败，请人工检查交易所仓位。"

    # ---------- 自动平仓 ----------
    def _handle_auto_exit(self, df_map: Dict[str, pd.DataFrame]):
        to_close = []

        for sym, pos in list(self.positions.items()):
            df5 = df_map.get(sym)
            if df5 is None or len(df5) < 5:
                continue

            last_ts = df5.index[-1]
            row = df5.iloc[-1]
            prev_row = df5.iloc[-2]

            # bars 计数
            if pos.last_bar_ts is None or last_ts > pos.last_bar_ts:
                pos.bars_held += 1
                pos.last_bar_ts = last_ts

            reason = None
            exit_price = None

            # 1) 时间止损
            if pos.max_bars > 0 and pos.bars_held >= pos.max_bars:
                reason = f"持仓 bars 超限 ({pos.bars_held} >= {pos.max_bars})"
                exit_price = float(row["close"])

            # 2) 触发 SL/TP
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
                else:
                    if high >= pos.stop_price:
                        reason = f"触发止损 {pos.stop_price:.4f}"
                        exit_price = pos.stop_price
                    elif low <= pos.take_price:
                        reason = f"触发止盈 {pos.take_price:.4f}"
                        exit_price = pos.take_price

            # 3) 趋势反转（反向信号）
            if reason is None or exit_price is None:
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

            # === 计算 PnL 并更新账户 ===
            print(
                f"[Exit] {sym} @ {last_ts}, side={pos.side}, "
                f"reason={reason}, exit_price={exit_price:.4f}"
            )

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
                # 更新逐币种 PnL 统计
                stats = self.pnl_stats[sym]
                if pnl >= 0:
                    stats["win"] += pnl
                else:
                    stats["loss"] += pnl
                stats["sum"] += pnl

            # === 实盘下平仓单 ===
            if self.cfg_live.live_mode and self.binance is not None and pos.qty > 0:
                side_str = "sell" if pos.side == "LONG" else "buy"
                close_params: Dict[str, object] = {"reduceOnly": True}
                if self.binance.hedge_mode:
                    close_params["positionSide"] = "LONG" if pos.side == "LONG" else "SHORT"

                # 将一次完整平仓流程视为一次下单动作（包含初次 + Failsafe）
                self.total_order_sent += 1
                success = False
                fail_reason = ""

                try:
                    order = self.binance.create_market_order(
                        symbol=sym,
                        side=side_str,
                        amount=pos.qty,
                        params=close_params,
                    )
                    if order is not None:
                        success = True
                except Exception as e:
                    print(f"[ExitOrder] {sym} 平仓下单异常: {e}")
                    order = None
                    fail_reason = str(e)

                if not success:
                    print(f"[ExitOrder] {sym} 平仓下单失败，启动 Failsafe 强制同步。")
                    # 触发 Failsafe：重新同步该 symbol 实盘仓位并尝试二次平仓
                    success, fs_reason = self._failsafe_force_close(sym, pos)
                    if not success and fs_reason:
                        fail_reason = fs_reason

                if success:
                    self.total_order_success += 1
                    self.last_fail_reason = ""
                    print(f"[ExitOrder] {sym} 平仓流程最终成功。")
                    to_close.append(sym)
                else:
                    self.total_order_fail += 1
                    self.last_fail_reason = (
                        f"{sym} 平仓最终失败: {fail_reason}" if fail_reason else f"{sym} 平仓最终失败（未知原因）"
                    )
                    print(f"[ExitOrder] {sym} 平仓流程最终失败，请人工检查。")
            else:
                # 非实盘模式 / 无 qty，仅在模型层平仓
                to_close.append(sym)

        for sym in to_close:
            self.positions.pop(sym, None)

    # ---------- 单轮刷新 ----------
    def run_one_cycle(self):
        self.refresh_count += 1
        cycle_signal_count = 0
        cycle_order_sent = 0
        cycle_order_success = 0
        cycle_order_fail = 0

        print(
            f"\n==== V31_15_AutoDetect_Failsafe 周期刷新 · symbols={self.cfg_live.symbols}, "
            f"topk={self.cfg_live.topk}, live={self.cfg_live.live_mode}, "
            f"refresh={self.cfg_live.refresh_seconds}s ===="
        )

        if self.cfg_live.live_mode and self.binance is None:
            print("[Engine] live_mode=True 但未提供 Binance 客户端，直接返回。")
            return

        df_map: Dict[str, pd.DataFrame] = {}
        score_map: Dict[str, float] = {}

        # 1) 拉取全部 K 线 & 计算趋势评分
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
                not in ["open", "high", "low", "close", "volume"]
            ]
            df5[ind_cols] = df5[ind_cols].ffill()

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

        # 3) 启动时同步账户持仓（只做一次）
        self._sync_positions_from_exchange(df_map)

        # 4) 自动平仓逻辑（时间止损 / SL / TP / 趋势反转）
        self._handle_auto_exit(df_map)

        # 5) TopK 中尝试开新仓
        for full_sym in topk_syms:
            df5 = df_map.get(full_sym)
            if df5 is None or len(df5) < 5:
                continue

            # 已有仓位则不加仓
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

            # 手续费（开仓）
            fee_entry = notional * self.core_cfg.fee_rate
            try:
                self.account.apply_pnl(-fee_entry, last_ts)
            except Exception as e:
                print(f"[Account] 应用开仓手续费时出错: {e}")

            # === 实盘下开仓单 ===
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
                    self.last_fail_reason = ""
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
                    self.last_fail_reason = f"{full_sym} 开仓失败（交易所未返回订单）"
            else:
                # 模拟模式：只记账，不真实下单
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

        # === 打印账户 & 统计信息 ===
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
        print("\n==== [Engine Status · V31_15_AutoDetect_Failsafe] ============================")
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

        # 输出逐币种 PnL 统计（只展示有交易的币）
        if self.pnl_stats:
            print("—— PnL 累计统计（按币种） ——")
            for sym, stats in self.pnl_stats.items():
                if stats["win"] == 0 and stats["loss"] == 0:
                    continue
                print(
                    f"{sym}: win={stats['win']:.2f}, loss={stats['loss']:.2f}, sum={stats['sum']:.2f}"
                )

        # 输出最近一次失败原因（如有）
        if self.last_fail_reason:
            print(f"最近一次下单失败原因: {self.last_fail_reason}")

        print("=========================================================")

    # ---------- 主循环 ----------
    def run_forever(self):
        print("==== 启动 V31_15_AutoDetect_Failsafe 多币轮动·Binance USDT-M Futures 实盘引擎（实时循环） ====")
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
# CLI & main
# ===========================
def parse_args():
    p = argparse.ArgumentParser(description="V31_15_AutoDetect_Failsafe 多币轮动 · Binance USDT-M Futures 实盘引擎")
    p.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,LTCUSDT,ADAUSDT,LINKUSDT,TONUSDT",
        help="监控币种列表，用逗号分隔",
    )
    p.add_argument("--topk", type=int, default=3, help="每轮选择最强 TopK 个币种")
    p.add_argument(
        "--leverage",
        type=float,
        default=3.0,
        help="杠杆倍数（供风控计算使用，不直接设置交易所杠杆）",
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
        help="强趋势 RR（默认 4.0）",
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
        help="每次刷新间隔秒数",
    )
    p.add_argument(
        "--initial-equity",
        type=float,
        default=10_000.0,
        help="初始资金（模型层），实盘模式下可被真实权益覆盖",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="启用实盘模式（连接 Binance USDT-M Futures）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    notifier = TelegramNotifier()
    notifier.send("🚀 V31 实盘系统启动成功")
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

    engine = LiveTradeEngineV31_15(cfg_live, binance=binance_client)
    engine.run_forever()


if __name__ == "__main__":
    main()
