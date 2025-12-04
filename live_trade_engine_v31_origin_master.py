# -*- coding: utf-8 -*-
"""
live_trade_engine_V31_OriginMaster.py

V31_OriginMaster · 多币轮动 · Binance USDT-M 永续合约 实盘引擎【注释教学版】

说明：
- 这是在 V31_13 基础上，增加了大量中文注释的“起源大师版”。
- 交易逻辑、风险控制、信号判定、下单流程等，全部保持与 V31_13 一致。
- 唯一的变化：加入了详尽的解释性注释，帮助你学习和二次开发。

核心功能：
1）多币种 5 分钟级别趋势轮动（TopK 最强币种参与）
2）V31_1 同款：ATR 止损 + RR 止盈 + trend_strength 动态调参
3）单币只持一个仓位（不加仓、不网格、不金字塔）
4）实盘支持 Binance USDT-M 永续，支持单向/对冲持仓
5）V31_13 新增：程序启动时自动接管交易所已有持仓（避免“裸单”）
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

# 从回测核心引擎中导入 V31_1 的关键组件
from v31_core_v31_1 import (
    V31CoreConfig,               # 回测版 V31 的参数配置对象（含杠杆、风险、费用等）
    WaterfallAccountV31_1,       # 资金账户类，负责管理权益、风险资金、抽水等（本版关闭抽水）
    build_multi_tf_indicators,   # 构建 5m/15m/1h 等多周期指标的函数
    entry_signal_v31,            # 核心入场信号函数（返回 LONG / SHORT / FLAT）
    compute_sl_tp_notional_v31,  # 根据 ATR / RR / 风险资金计算止损、止盈、名义仓位、最长持仓 bars
)


# ==============================
# 配置加载
# ==============================
def load_live_config(path: str = "config_live_v31.yaml") -> dict:
    """
    从 YAML 文件中加载交易所配置、API Key 等信息。

    path:
        配置文件路径，默认使用项目中约定的 config_live_v31.yaml

    返回值：
        dict 格式的配置字典，其中应包含：
        - binance: { apiKey, secret, enable_trading, hedge_mode, options, proxies, ... }
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
    对 ccxt.binance 进行一层薄封装，专门用于 USDT-M 永续合约。

    主要职责：
    - 读取 YAML 中的配置，初始化 ccxt 客户端
    - 统一处理 proxies、options、enable_trading 等
    - 提供行情接口：fetch_ohlcv_5m
    - 提供账户接口：fetch_futures_equity_usdt
    - 提供下单接口：create_market_order
    - 自动或手动识别 hedge_mode（单向/对冲持仓模式）

    注意：
    - enable_trading = False 时，即使传入 --live 也只会“打印即将下单”，不会真正下单。
    """

    def __init__(self, cfg: dict):
        """
        期望 YAML 结构示例：

        binance:
          name: binance
          apiKey: "..."
          secret: "..."
          enable_trading: true         # 是否允许真实下单
          hedge_mode: true             # 可选，若写了则强制使用该持仓模式
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
            # 没有 API Key 和 Secret，就没有办法实盘
            raise ValueError("Binance 配置缺少 apiKey / secret")

        # 是否真实下单：
        # - True：会调用 create_order，真的发到交易所
        # - False：只打印“准备下单”，用于干跑测试 / 回测风格验证
        self.enable_trading: bool = bool(ex_cfg.get("enable_trading", False))

        # 网络代理设置（可选）：
        # - 如果在国内服务器上跑，通常需要配置 https 代理才能访问 Binance
        proxies = ex_cfg.get("proxies") or None
        # options：ccxt 的额外参数（如 defaultType=future，adjustForTimeDifference 等）
        options = ex_cfg.get("options") or {}

        # other：将除了通用字段以外的配置透传给 ccxt（如 timeout、recvWindow 等）
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
                "enableRateLimit": True,  # 自动限速，避免触发交易所限流
                "proxies": proxies,
                "options": options,
                **other,
            }
        )

        # 1) 优先：查看 YAML 是否显式指定 hedge_mode
        #    - True：对冲模式（dual side，LONG/SHORT 分开持仓）
        #    - False：单向持仓（one-way，只有 BOTH）
        explicit_hedge = ex_cfg.get("hedge_mode")
        if explicit_hedge is not None:
            self.hedge_mode = bool(explicit_hedge)
            mode_str = "对冲模式(双向持仓)" if self.hedge_mode else "单向持仓模式"
            print(f"[Binance] 配置文件中显式指定 hedge_mode={self.hedge_mode} → {mode_str}")
            return

        # 2) 否则：尝试通过 fetch_balance 自动识别账户当前持仓模式
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
                # 没有 positionSide 字段，一般是单向持仓（BOTH），保守按单向处理
                self.hedge_mode = False
                print(
                    "[Binance] 无 positionSide 字段信息，默认按单向持仓模式处理 "
                    f"(sides={sides})"
                )
            else:
                # 典型情况：
                # - 对冲模式：可能出现 LONG / SHORT
                # - 单向模式：只有 BOTH
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
            # 如果探测失败（API版本/权限等问题），就默认单向持仓，避免误用 LONG/SHORT 参数
            self.hedge_mode = False
            print(f"[Binance] 警告：无法自动检测持仓模式，默认按单向持仓模式处理。err={e}")

    # ---- 行情 ----
    def fetch_ohlcv_5m(self, symbol: str, limit: int = 500) -> Optional[pd.DataFrame]:
        """
        固定拉取最近 limit 根 5m K 线，默认 500 根（约 500*5min ≈ 41.7 小时）。

        为什么不用 since？
        - 使用 since 容易只拿到 1~2 根 K 线，指标不完整。
        - 直接用 limit=500 拿最近一段，既简单又稳。

        返回：
        - DataFrame，index 为 UTC 时间戳，列包括 open/high/low/close/volume。
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

    # ---- 账户（获取 USDT 权益）----
    def fetch_futures_equity_usdt(self) -> Optional[float]:
        """
        获取 USDT-M 永续账户 USDT 总权益（equity，含未实现盈亏）。
        仅用于：
        - 启动时“参考”初始资金设置（account.initial_capital）
        - 打印监控，帮助你对照模型账户和实盘账户

        注意：
        - 这里不会影响实盘账户的真实资金，只是读。
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
                # 有些 ccxt 版本结构略不同，这里再做一次兼容处理
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
        创建市价单（market order）。

        参数：
        - symbol: 如 "BTCUSDT"
        - side  : "buy" / "sell"
        - amount: 下单数量（合约张数，Binance USDT-M 中即为币的数量）
        - params: 额外参数 dict
            * 对冲模式(hedge_mode=True) 时必须带上 positionSide: "LONG"/"SHORT"
            * reduceOnly=True 表示“只减仓不增仓”（平仓用）

        逻辑：
        - 若 enable_trading=False，仅打印，不真正下单（用于安全演练）
        - 若 enable_trading=True，调用 ccxt.create_order 真正向交易所发单
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
# 引擎配置（实盘层面的参数）
# ==============================
@dataclass
class LiveEngineConfig:
    """
    实盘引擎参数（与 V31 回测参数相关联，但更加简化）：

    symbols:
        要监控的交易对列表，例如 ["BTCUSDT", "ETHUSDT", ...]

    timeframe:
        当前版本固定用 "5m"，这里主要是留作未来扩展接口。

    topk:
        每一轮选择“趋势评分最高”的前 k 个币种参与交易。
        - 越大：多币同时持仓可能性越高，收益机会多，风险也更分散但总风险更大
        - 越小：集中在少数几个最强趋势上，更纯粹的 Alpha，但可能错过其他币种行情

    leverage:
        给回测/核心逻辑使用的“参考杠杆”（如 3x、5x）：
        - 用于计算止损宽度、风险等（compute_sl_tp_notional_v31 内部会用到）
        - 并不会直接传给 Binance（Binance 实际杠杆由交易所设置上限决定）

    risk_per_trade:
        单笔交易风险占总资金的比例，如 0.01 = 1%：
        - 越大：每笔亏损更重，收益也更快，曲线更陡峭，回撤放大
        - 越小：更稳健，但收益曲线偏平

    refresh_seconds:
        行情刷新周期（秒）。
        - 太小：更及时但更消耗 API / CPU
        - 太大：信号响应更慢，有可能错过最优入场点

    initial_equity:
        初始资金（模型层）。
        - 若在实盘模式下能读到交易所 USDT 权益，则会用实际权益覆盖此值。

    live_mode:
        是否连接实盘：
        - False：仅模型层运算（不调用交易所 API 下单）
        - True ：会使用 BinanceFuturesV31，下单行为取决于 config_live_v31.yaml 中的 enable_trading

    rr_strong / rr_normal:
        强趋势/普通趋势下的止盈 RR（Risk:Reward 比例）
        - RR=4 表示：单笔理想盈利 = 4 * 单笔风险

    sl_mult_strong / sl_mult_normal:
        强趋势/普通趋势下的 ATR 止损倍数：
        - 越大：止损更宽，容忍更大的波动，回撤更深但更容易 hold 住趋势
        - 越小：止损紧，不容易大亏，但可能更容易被震荡洗掉
    """
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
# 实盘持仓结构体
# ==============================

@dataclass
class LivePositionV31:
    """
    单笔持仓的信息（实盘 + 模型层共用）：

    symbol:
        交易对，如 "BTCUSDT"

    side:
        "LONG" 或 "SHORT"

    entry_ts:
        开仓时刻（K 线时间）

    entry_price:
        开仓价格（包含滑点）

    qty:
        实际下单数量（contracts），平仓时会用它去 reduceOnly 或平掉仓位。

    notional:
        名义仓位（美元），用于：
        - 计算 PnL
        - 计算手续费
        - 计算风险资金占用

    stop_price / take_price:
        固定止损价 / 固定止盈价
        - 由 compute_sl_tp_notional_v31 基于 ATR + RR 等动态算出

    max_bars:
        最大持仓 bar 数（5m bar 数量）
        - 超过这个 bar 数，无论盈亏如何都会强制平仓（时间止损机制）

    bars_held:
        已持有 bar 数

    last_bar_ts:
        最近一次更新 bars_held 的 K 线时间戳，用于避免重复计数
    """
    symbol: str
    side: str  # "LONG" or "SHORT"
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
# 多币轮动实盘引擎 · V31_OriginMaster
# ==============================

class LiveTradeEngineV31_13:
    """
    实盘引擎核心类（教学版名字是 V31_OriginMaster，内部打印仍保持 V31_13 以兼容之前日志）：

    主要职责：
    - 管理多币 K 线拉取和指标计算
    - V31 风格趋势评分 + TopK 轮动
    - 生成 V31_1 同款入场信号
    - 计算 SL/TP/名义仓位/最大持仓时间
    - 管理实时持仓、自动平仓（止盈、止损、趋势反转、时间止损）
    - 记录模型账户资金变化 & 实盘对照
    - 启动时自动接管 Binance 账户中现有持仓（避免裸单）
    """

    def __init__(
        self,
        cfg_live: LiveEngineConfig,
        binance: Optional[BinanceFuturesV31] = None,
    ):
        self.cfg_live = cfg_live
        self.binance = binance

        # 创建 V31_1 回测核心配置对象
        # 注意：这里 symbol 传 "MULTI" 仅作为占位，不影响逻辑
        self.core_cfg = V31CoreConfig(
            symbol="MULTI",
            days=365,
            leverage=cfg_live.leverage,
            risk_per_trade=cfg_live.risk_per_trade,
        )

        # 将 LIVE 端的 RR / SL 参数同步到核心配置（与回测逻辑对齐）
        self.core_cfg.rr_strong = cfg_live.rr_strong
        self.core_cfg.rr_normal = cfg_live.rr_normal
        self.core_cfg.sl_mult_strong = cfg_live.sl_mult_strong
        self.core_cfg.sl_mult_normal = cfg_live.sl_mult_normal

        # 资金账户：
        # - initial_capital: 初始资金（会被 Binance 实盘权益覆盖）
        # - withdraw_rate / growth_threshold / enable_waterfall 在此版本中关闭抽水功能
        self.account = WaterfallAccountV31_1(
            initial_capital=cfg_live.initial_equity,
            withdraw_rate=0.0,        # 不做抽水
            growth_threshold=0.01,    # 不重要，因为 enable_waterfall=False
            enable_waterfall=False,   # 关闭瀑布抽水
        )

        # 若 live_mode=True 且提供了 Binance 客户端，则尝试用“实盘 USDT 权益”覆盖初始资金
        if self.cfg_live.live_mode and self.binance is not None:
            eq = self.binance.fetch_futures_equity_usdt()
            if eq is not None and eq > 0:
                self.account.initial_capital = float(eq)
                self.account.trading_capital = float(eq)
                self.account.last_high = float(eq)
                print(f"[Account] 使用 Binance 实盘 USDT 权益作为初始资金: {eq:.2f}")
            else:
                # 如果获取失败，就沿用命令行传入的 initial_equity
                print(
                    f"[Account] 未能获取实盘权益，继续使用配置中的 initial_equity={cfg_live.initial_equity:.2f}"
                )

        # 记录连续亏损次数（以 symbol 为粒度）
        # - 用于 entry_signal_v31 内部做“连亏保护/适度收缩风险”等逻辑
        self.consecutive_losses: Dict[str, int] = {s: 0 for s in cfg_live.symbols}

        # 实盘持仓字典：
        # - key: symbol (例如 "BTCUSDT")
        # - value: LivePositionV31
        self.positions: Dict[str, LivePositionV31] = {}

        # 标记：是否已经从交易所同步过一次持仓
        self.synced_positions_from_exchange: bool = False

        # 运行统计
        self.start_time = pd.Timestamp.utcnow()
        self.refresh_count: int = 0          # 刷新次数（run_one_cycle 调用次数）
        self.total_signal_count: int = 0     # 累计产生的开仓信号数
        self.total_order_sent: int = 0       # 累计尝试下单次数
        self.total_order_success: int = 0    # 累计下单成功次数
        self.total_order_fail: int = 0       # 累计下单失败次数

    # ---------- 趋势评分 ----------
    def _calc_trend_score(self, df5: pd.DataFrame) -> float:
        """
        对单一币种的 5m 指标数据进行打分，用于多币轮动时排序。

        评分逻辑：
        - 使用最近约 24 小时的数据（12*24 根 5m K 线）
        - 主要考虑：
            * 收盘价涨跌幅（ret）
            * trend_strength 平均值
            * trend_dir 平均值

        score = ret*100 + strength*5 + dir_mean*2

        影响：
        - score 越大 → 越有可能进入 TopK，越优先参与交易
        - 该函数只影响“选币顺序”，不影响单币的入场/止盈/止损逻辑
        """
        if df5 is None or df5.empty or len(df5) < 60:
            return 0.0

        recent = df5.iloc[-12 * 24 :]  # 最近约 24 小时（12 根/小时 * 24 小时）
        if recent.empty:
            return 0.0

        close = recent["close"]
        ret = float(close.iloc[-1] / close.iloc[0] - 1.0)

        strength = float(recent["trend_strength"].mean())
        dir_mean = float(recent["trend_dir"].mean())

        # 这里系数是经验值，调大调小会改变各因子权重，不建议轻易乱改
        score = ret * 100.0 + strength * 5.0 + dir_mean * 2.0
        return score

    # ---------- 启动后从交易所同步当前未平仓位 ----------
    def _sync_positions_from_exchange(self, df_map: Dict[str, pd.DataFrame]):
        """
        仅在 live_mode 且尚未同步时执行一次：
        - 从 Binance 期货账户中读取当前所有未平仓位
        - 对于 symbol 在本次监控列表中的仓位，生成 LivePositionV31 写入 self.positions
        - 目的：让程序重启后可以“接管”原有实盘仓位，继续管理 SL/TP/max_bars 等风控

        同步规则：
        1) 只同步本次监控 symbol 列表内的仓位
        2) 只同步 positionAmt != 0 的币种（有真实持仓）
        3) 使用交易所的 entryPrice 和 positionAmt 构造 notional
        4) 利用最新一根 K 线的 atr_1h 和 trend_strength 调用 compute_sl_tp_notional_v31
           重新生成 stop_price、take_price、max_bars，并写入 engine.positions
        """

        if self.synced_positions_from_exchange:
            # 已同步过一次，就不再重复执行
            return
        if not self.cfg_live.live_mode or self.binance is None:
            # 非实盘模式或没有交易所客户端，不必同步
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

        # 将 cfg_live.symbols 转成一组 normalized 的 symbol（大写+补 USDT）
        symbols_set = {
            s.upper() if s.upper().endswith("USDT") else (s.upper() + "USDT")
            for s in self.cfg_live.symbols
        }

        synced_count = 0
        for p in positions:
            symbol = str(p.get("symbol") or "").upper()
            if symbol not in symbols_set:
                continue

            pos_amt = float(p.get("positionAmt") or 0.0)
            if abs(pos_amt) < 1e-8:
                # 无持仓
                continue

            entry_price = float(p.get("entryPrice") or 0.0)
            if entry_price <= 0:
                continue

            # side 判定：
            # - pos_amt > 0 → LONG
            # - pos_amt < 0 → SHORT
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

            # 重新计算一套 stop_price / take_price / max_bars：
            # - 这里用 account.risk_capital() 只是为了调用接口，其返回的 notional 在这里不采用
            try:
                stop_price, take_price, _, max_bars, _, rr_used = compute_sl_tp_notional_v31(
                    cfg=self.core_cfg,
                    side=side,
                    trend_strength=trend_strength,
                    entry_price=entry_price,
                    atr_1h=atr_1h,
                    equity_for_risk=self.account.risk_capital(),
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
                entry_ts=last_ts,          # 作为接管时刻的参考时间
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
        """
        对当前所有持仓进行“自动平仓检测”：

        平仓触发条件（任一满足即平仓）：
        1）时间止损：bars_held >= max_bars
        2）触发止损价：K 线 high/low 穿越 stop_price
        3）触发止盈价：K 线 high/low 穿越 take_price
        4）趋势反转：entry_signal_v31 给出与当前持仓相反方向的信号

        平仓步骤：
        - 计算 PnL（含手续费、滑点）并更新 account
        - 调整 consecutive_losses
        - 在实盘模式下，发 reduceOnly 市价单平仓
        """
        to_close = []

        for sym, pos in list(self.positions.items()):
            df5 = df_map.get(sym)
            if df5 is None or len(df5) < 5:
                continue

            last_ts = df5.index[-1]
            row = df5.iloc[-1]
            prev_row = df5.iloc[-2]

            # 更新持仓 bar 计数：
            # - 仅当出现新 K 线时才 +1（避免一个 bar 内多次重复计算）
            if pos.last_bar_ts is None or last_ts > pos.last_bar_ts:
                pos.bars_held += 1
                pos.last_bar_ts = last_ts

            reason = None
            exit_price = None

            # 1) 最大持仓 bars 约束：超过 max_bars 强制离场（时间止损）
            if pos.max_bars > 0 and pos.bars_held >= pos.max_bars:
                reason = f"持仓 bars 超限 ({pos.bars_held} >= {pos.max_bars})"
                exit_price = float(row["close"])

            # 2) 基于 K 线高低价的 TP / SL 判定：
            #    - 使用 high/low 而不是 close，可以在 K 内部触发止损/止盈
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

            # 3) 趋势反转：
            #    - 使用 entry_signal_v31 再次计算当前 bar 的信号
            #    - 如果给出多空方向与当前持仓相反，则触发“趋势反转平仓”
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

            # 若没有任何理由触发平仓，则继续持有
            if reason is None or exit_price is None:
                continue

            print(f"[Exit] {sym} @ {last_ts}, side={pos.side}, reason={reason}, exit_price={exit_price:.4f}")

            # === 先计算 PnL 并更新账户 ===
            notional = pos.notional
            if notional > 0:
                # price_change_pct：价格变动占 entry_price 的比例
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

                # 连续亏损计数调整：
                # - 若 pnl <= 0 → 连亏+1
                # - 若 pnl > 0  → 清零
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
                # 与开仓方向相反的方向平仓：
                # - 原来 LONG → 平仓要卖 "sell"
                # - 原来 SHORT → 平仓要买 "buy"
                side_str = "sell" if pos.side == "LONG" else "buy"
                close_params: Dict[str, object] = {
                    "reduceOnly": True,  # 只减仓，不允许增加仓位，防止方向弄反时加仓
                }
                if self.binance.hedge_mode:
                    # 对冲模式时必须声明 positionSide，否则可能得到 -4061 错误
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

        # 将已经离场的 symbol 从持仓 dict 中删除
        for sym in to_close:
            self.positions.pop(sym, None)

    # ---------- 单轮计算 / 信号生成 / 下单 ----------
    def run_one_cycle(self):
        """
        单次刷新周期：

        流程：
        1）拉取所有监控币种的 5m K 线 + 指标
        2）计算 trend_score，并选出 TopK 币种
        3）首次运行时，同步 Binance 现有仓位（接管）
        4）对现有持仓执行自动平仓逻辑（SL/TP/趋势反转/时间止损）
        5）对 TopK 且当前无持仓的币种，计算入场信号并决定是否开仓
        6）打印账户及运行状态
        """
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

            # build_multi_tf_indicators 会构建多周期指标，返回一个 context dict
            # 我们这里仅使用其中 "df_5m"（5 分钟数据）
            ctx = build_multi_tf_indicators(df_raw, self.core_cfg)
            df5 = ctx["df_5m"].copy()

            # 对指标列做 ffill，避免中间 NaN 导致信号判断异常
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

        # 对所有币种按 score 排序，取 topk 个参与
        sorted_syms = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        topk_syms = [s for s, _ in sorted_syms[: self.cfg_live.topk]]
        scores_str = ", ".join([f"{k}: {v:.6f}" for k, v in score_map.items()])
        print(f"[TopK] 当前最强: {topk_syms}, scores={{ {scores_str} }}")

        # 2) 若尚未从交易所同步实盘持仓，则先同步一次（只执行一次）
        self._sync_positions_from_exchange(df_map)

        # 3) 对现有持仓执行自动平仓逻辑
        self._handle_auto_exit(df_map)

        # 4) 对 TopK 中当前尚未持仓的币种，计算入场信号并尝试开仓
        for full_sym in topk_syms:
            df5 = df_map.get(full_sym)
            if df5 is None or len(df5) < 5:
                continue

            # 若当前已经持仓，则不重复开仓（单币只持一个仓位的约束）
            if full_sym in self.positions:
                continue

            last_ts = df5.index[-1]
            row = df5.iloc[-1]
            prev_row = df5.iloc[-2]

            # 利用 V31 的入场信号函数决定是否开多/开空/空仓
            side, trade_type = entry_signal_v31(
                cfg=self.core_cfg,
                consecutive_losses=self.consecutive_losses.get(full_sym, 0),
                ts=last_ts,
                row=row,
                prev_row=prev_row,
            )
            if side == "FLAT" or trade_type is None:
                # 没有买卖信号，则不进场
                continue

            atr_1h = float(row.get("atr_1h", 0.0))
            if atr_1h <= 0:
                # 没有 ATR，无法计算止损 & 仓位大小
                continue

            price_close = float(row["close"])
            # 加滑点：
            # - LONG：按略高于 close 的价格入场（买贵一点）
            # - SHORT：按略低于 close 的价格入场（卖便宜一点）
            if side == "LONG":
                entry_price = price_close * (1.0 + self.core_cfg.slippage)
            else:
                entry_price = price_close * (1.0 - self.core_cfg.slippage)

            # equity_for_risk：用于控制单笔风险金额：
            # - 一般为 trading_capital（可用风险资金）
            # - 随资金曲线变化动态调整
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

            # 若任何关键参数不合法，则跳过该信号
            if notional <= 0 or stop_price <= 0 or take_price <= 0 or max_bars <= 0:
                continue

            # qty：要下单的数量 = 名义仓位 / 价格
            qty = notional / entry_price
            direction_str = "多" if side == "LONG" else "空"
            print(
                f"[Signal] {full_sym} @ {last_ts}, 开{direction_str}, "
                f"notional={notional:.2f}, qty={qty:.6f}, entry≈{entry_price:.4f}, "
                f"SL={stop_price:.4f}, TP={take_price:.4f}, RR={rr_used:.2f}"
            )

            self.total_signal_count += 1
            cycle_signal_count += 1

            # 开仓手续费：
            # - 先从账户中扣除，用于模拟资金曲线真实表现
            fee_entry = notional * self.core_cfg.fee_rate
            try:
                self.account.apply_pnl(-fee_entry, last_ts)
            except Exception as e:
                print(f"[Account] 应用开仓手续费时出错: {e}")

            # === 实盘开仓 ===
            if self.cfg_live.live_mode and self.binance is not None:
                side_str = "buy" if side == "LONG" else "sell"
                amount = qty

                open_params: Dict[str, object] = {}
                if self.binance.hedge_mode:
                    # 对冲模式：必须带上 positionSide 告诉交易所是 LONG 还是 SHORT
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
                    # 记录持仓，供后续自动平仓逻辑使用
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
                # 非实盘模式（回测/干跑）：只在模型层记录持仓
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

        # 5) 打印模型账户资金情况 + 实盘对照
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

        # 打印本轮运行状态
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
        """
        无限循环运行引擎：

        - 每次循环调用 run_one_cycle()
        - 中间 sleep(refresh_seconds)
        - 支持 Ctrl+C 手动中断
        """
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
    """
    命令行参数解析：

    常用参数说明（重点）：

    --symbols
        监控币种列表，逗号分隔。
        例如：BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT...
        - 币种越多，机会越多，但每轮计算量也越大。

    --topk
        每一轮从所有监控币种中选出“趋势评分最高”的前 k 个，允许参与入场。
        - 控制“最大同时持仓币种数量”的上限（单币只持一单）。
        - 越大：持仓更分散，整体风险更大；越小：更集中，但可能错过部分机会。

    --leverage
        给 V31 核心配置使用的杠杆参数：
        - 影响止损宽度、风险、名义仓位计算
        - 不会直接传给 Binance，Binance 实际杠杆由交易所规则决定（如 BNB 15x、SOL 20x）。

    --risk-per-trade
        单笔风险占总资金的比例，如 0.01 表示 1%：
        - 账户权益 5000USDT 时，每笔最多亏损 ≈ 50USDT。
        - 越大：收益/回撤都放大；越小：更“佛系”，风险更低。

    --rr-strong / --rr-normal
        强趋势 / 普通趋势下使用的 RR：
        - RR=4 表示：理论止盈点约为止损距离的 4 倍。
        - 越大：更加追求“大利润单”，但命中率可能略降低。

    --sl-mult-strong / --sl-mult-normal
        强趋势 / 普通趋势下的 ATR 止损倍数：
        - 越大：止损更宽，容错性高，但单笔风险区间变大。
        - 越小：更容易被震荡洗掉，但能快速止损。

    --refresh-seconds
        每轮刷新间隔：
        - 不建议小于 5 秒，否则可能频繁打 API。
        - 10 秒 对于 5m K 线是比较舒服的配置。

    --initial-equity
        初始资金（模型账户用）：
        - 若启用 --live 且能成功读取 Binance USDT 权益，会使用真实权益覆盖此值。

    --live
        是否启用实盘模式：
        - 若未加 --live，全程不下单，只做模型模拟。
        - 加了 --live，但 config_live_v31.yaml 中 enable_trading=false → 仍不真实下单。
        - 真正实盘需：--live + enable_trading=true + 正确的 API Key/Secret。
    """
    p = argparse.ArgumentParser(description="V31_13 多币轮动 · Binance USDT-M Futures 实时引擎（V31_OriginMaster 注释版）")
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
        help="杠杆倍数（默认 3.0，与 V31_1 回测配置一致，仅参与风控计算，不直接设置交易所杠杆）",
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
