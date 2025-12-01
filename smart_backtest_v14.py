# smart_backtest_v14.py
# V14: AI 自适应权重 + 强化 Regime 引擎版
#
# 使用方式：
#   python smart_backtest_v14.py --symbols "BTC/USDT,ETH/USDT" --days 60 --data-source local
#
# 依赖：
#   - local_data_engine.py       -> LocalDataEngine
#   - real_market_data_v3.py    -> RealMarketData（如果用 real 模式）
#   - real_strategies.py        -> basic trend signals（可选，没有就用内置简单策略）

import argparse
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from local_data_engine import LocalDataEngine
from real_market_data_v3 import RealMarketData

logger = logging.getLogger("SmartBacktestV14")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ========= 工具函数 =========

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(period).mean()
    return atr


def compute_ma(df: pd.DataFrame, period: int) -> pd.Series:
    return df["close"].rolling(period).mean()


def slope(series: pd.Series, window: int = 10) -> pd.Series:
    # 简单近似：当前值 - N bars 前
    return (series - series.shift(window)) / window


# ========= Regime & 权重 引擎 =========

@dataclass
class RegimeInfo:
    trend_score: float = 0.0   # 趋势强度 0~1
    range_score: float = 0.0   # 震荡强度 0~1
    vol_score: float = 0.0     # 波动质量评分 0~1
    regime: str = "unknown"    # "trend", "range", "mixed"


@dataclass
class SymbolState:
    name: str
    equity: float
    max_equity: float
    max_drawdown: float = 0.0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    cold_streak: int = 0
    cooldown_until: int = -1  # 冷静期结束 bar 索引（基于 LTF 索引）
    pnl_history: List[float] = field(default_factory=list)


class AIWeightEngine:
    """
    简易 AI 权重引擎：
    - 根据 Regime 信息 + 近端 pnl_history 给出该币种的权重 0~1
    - 再归一化到所有币种
    """
    def __init__(self, symbols: List[str]):
        self.symbols = symbols

    def score_symbol(self, regime: RegimeInfo, state: SymbolState) -> float:
        # 趋势越强，评分越高；震荡越强，评分越低
        trend_part = regime.trend_score
        range_penalty = 1.0 - regime.range_score

        # 近期 pnl 趋势（最后 20 笔）
        if state.pnl_history:
            recent = state.pnl_history[-20:]
            pnl_avg = np.mean(recent)
            pnl_sign = np.tanh(pnl_avg / (state.equity + 1e-9))
        else:
            pnl_sign = 0.0

        # 综合评分：0~1 区间
        raw = 0.5 * trend_part + 0.3 * range_penalty + 0.2 * (pnl_sign + 1) / 2
        return max(0.0, min(1.0, raw))

    def compute_weights(self, regimes: Dict[str, RegimeInfo],
                        states: Dict[str, SymbolState]) -> Dict[str, float]:
        scores = {}
        for sym in self.symbols:
            r = regimes.get(sym)
            s = states.get(sym)
            if r is None or s is None:
                scores[sym] = 0.0
            else:
                scores[sym] = self.score_symbol(r, s)

        total = sum(scores.values())
        if total <= 1e-9:
            # 所有都很差 → 均分一点小权重
            n = len(self.symbols)
            return {sym: 1.0 / n for sym in self.symbols}

        return {sym: v / total for sym, v in scores.items()}


# ========= 核心回测引擎 =========

class SmartBacktestV14:
    def __init__(
        self,
        symbols: List[str],
        initial_capital: float = 10000,
        rr_min: float = 1.5,
        loss_cooldown_n: int = 3,
    ):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.rr_min = rr_min
        self.loss_cooldown_n = loss_cooldown_n

        self.states: Dict[str, SymbolState] = {
            sym: SymbolState(name=sym, equity=initial_capital / len(symbols),
                             max_equity=initial_capital / len(symbols))
            for sym in symbols
        }

        self.weight_engine = AIWeightEngine(symbols)

    def _detect_regime(
        self,
        df_ltf: pd.DataFrame,
        df_mtf: pd.DataFrame,
        df_htf: pd.DataFrame,
    ) -> RegimeInfo:
        """
        用多周期 MA 斜率 + ATR 收缩/扩张 + BOLL 带宽 简易做 Regime 判别
        """

        # 使用最后一段窗口
        ltf_tail = df_ltf.tail(200)
        mtf_tail = df_mtf.tail(200)
        htf_tail = df_htf.tail(200)

        # 趋势：使用 1h & 4h MA 斜率
        ma_mtf_fast = compute_ma(mtf_tail, 20)
        ma_mtf_slow = compute_ma(mtf_tail, 60)
        ma_htf_fast = compute_ma(htf_tail, 20)
        ma_htf_slow = compute_ma(htf_tail, 60)

        mtf_slope = slope(ma_mtf_fast, 10).iloc[-1]
        htf_slope = slope(ma_htf_fast, 5).iloc[-1]

        # 标准化后做成 0~1
        trend_raw = abs(mtf_slope) + abs(htf_slope)
        trend_score = np.tanh(trend_raw * 500)  # 放大一点斜率敏感度

        # 震荡：BOLL 带宽 + ATR 收缩
        close_ltf = ltf_tail["close"]
        ma_ltf = close_ltf.rolling(20).mean()
        std_ltf = close_ltf.rolling(20).std()
        boll_width = (std_ltf / ma_ltf).iloc[-1]  # 相对波动宽度

        atr_ltf = compute_atr(ltf_tail, 14)
        atr_ratio = (atr_ltf / close_ltf).iloc[-1]

        # 如果波动宽度中等 + ATR 一般 → 更偏震荡
        bw_norm = np.tanh((boll_width * 200))
        atr_norm = np.tanh((atr_ratio * 200))

        range_score = (bw_norm + (1 - atr_norm)) / 2
        range_score = max(0.0, min(1.0, range_score))

        # 波动质量：过低 or 过高都不好
        vol_score = 1.0 - abs(atr_norm - 0.3)  # 偏 0.3 比较舒服
        vol_score = max(0.0, min(1.0, vol_score))

        # Regime 决策
        if trend_score > 0.6 and range_score < 0.5:
            regime = "trend"
        elif range_score > 0.6 and trend_score < 0.4:
            regime = "range"
        else:
            regime = "mixed"

        return RegimeInfo(
            trend_score=float(trend_score),
            range_score=float(range_score),
            vol_score=float(vol_score),
            regime=regime,
        )

    def _generate_signal_row(
        self,
        row_ltf: pd.Series,
        trend_regime: RegimeInfo,
    ) -> int:
        """
        方向决策：
        - 只在 regime.trend 或 mixed 且 trend_score > 某阈值时开单
        - 简单用 MA 快慢线 + close 相对 MA 位置 做方向
        return: 1=多头, -1=空头, 0=观望
        """
        # 简化：只做多头逻辑（crypto 长期向上，空头容易被嘎）
        if trend_regime.regime == "range" and trend_regime.trend_score < 0.4:
            return 0

        # row_ltf 已包含 ma_fast/ma_slow 等字段时可用，否则用 close vs ma
        close = row_ltf["close"]
        ma_fast = row_ltf.get("ma_fast", np.nan)
        ma_slow = row_ltf.get("ma_slow", np.nan)

        if np.isnan(ma_fast) or np.isnan(ma_slow):
            return 0

        # 简单趋势多头：快线在慢线上 + close 在快线上方
        if ma_fast > ma_slow and close > ma_fast:
            return 1

        return 0

    def _apply_trade_logic(
        self,
        sym: str,
        df_ltf: pd.DataFrame,
        regime: RegimeInfo,
        symbol_weight: float,
        state: SymbolState,
    ) -> Tuple[float, int, int, float]:
        """
        单币种交易循环：
        - 只做多头
        - 进场：方向信号 = 1
        - 止损：ATR * 1.2
        - 止盈：ATR * 2.5，RR ~ 2+
        - 仓位：基于 symbol_weight & Regime.vol_score 自适应
        """

        equity = state.equity
        max_equity = state.max_equity
        max_dd = state.max_drawdown
        trades = state.trades
        wins = state.wins
        losses = state.losses
        cold_streak = state.cold_streak
        cooldown_until = state.cooldown_until

        position = 0.0
        entry_price = 0.0
        atr = compute_atr(df_ltf, 14)
        df = df_ltf.copy()
        # 为方向判断补充 MA
        df["ma_fast"] = compute_ma(df, 20)
        df["ma_slow"] = compute_ma(df, 60)

        # 仓位因子：Regime 越趋势 + 波动质量越好 → 仓位越高
        regime_factor = 0.5 * regime.trend_score + 0.5 * regime.vol_score
        regime_factor = 0.3 + 0.7 * regime_factor   # 最低 0.3，最高 1.0

        # 最终仓位比例（相对这个 symbol 的 equity）
        position_scale = symbol_weight * regime_factor  # 0~1 左右

        for i, (idx, row) in enumerate(df.iterrows()):
            price = row["close"]
            this_atr = atr.iloc[i]

            # 冷静期检查
            if cooldown_until >= 0 and i < cooldown_until:
                # 不开新仓，只维护旧仓止损/止盈
                pass
            else:
                # 不在冷静期，可生成新信号
                direction = self._generate_signal_row(row, regime)

                if position == 0 and direction == 1 and this_atr > 0:
                    # 进场：按 ATR 控制风险，每笔风险约 1% equity
                    risk_per_trade = equity * 0.01
                    qty = (risk_per_trade / (this_atr * 1.2)) * position_scale
                    if qty > 0:
                        position = qty
                        entry_price = price
                        # 记录（不立刻计 pnl）
                        continue

            # 仓位管理：有仓位时检查止损/止盈
            if position > 0 and this_atr > 0:
                stop_loss = entry_price - this_atr * 1.2
                take_profit = entry_price + this_atr * 2.5

                # 触发止损
                if price <= stop_loss:
                    pnl = (price - entry_price) * position
                    equity += pnl
                    trades += 1
                    losses += 1
                    cold_streak += 1
                    state.pnl_history.append(pnl)

                    # 更新冷静期：连续亏损过多时进入冷静
                    if cold_streak >= self.loss_cooldown_n:
                        # 冷静期长度与 ATR/价格有关（波动大就再冷静一下）
                        cool_len = int(144 * (1 + regime.range_score))
                        cooldown_until = i + cool_len
                        logger.info(
                            "🧊 %s 连续亏损(%d) → 冷静期 %d bars",
                            sym, cold_streak, cool_len,
                        )

                    position = 0
                    entry_price = 0
                # 触发止盈
                elif price >= take_profit:
                    pnl = (price - entry_price) * position
                    equity += pnl
                    trades += 1
                    wins += 1
                    cold_streak = 0
                    state.pnl_history.append(pnl)
                    position = 0
                    entry_price = 0

            # 更新最大权益 & 回撤
            if equity > max_equity:
                max_equity = equity
            dd = (equity - max_equity) / max_equity
            if dd < max_dd:
                max_dd = dd

        # 回写状态
        state.equity = equity
        state.max_equity = max_equity
        state.max_drawdown = max_dd
        state.trades = trades
        state.wins = wins
        state.losses = losses
        state.cold_streak = cold_streak
        state.cooldown_until = cooldown_until

        return equity, trades, wins, max_dd

    # ---------- 对外接口 ----------
    def run_symbol(
        self,
        sym: str,
        df_ltf: pd.DataFrame,
        df_mtf: pd.DataFrame,
        df_htf: pd.DataFrame,
    ) -> Dict:
        # 1) Regime 分析
        regime = self._detect_regime(df_ltf, df_mtf, df_htf)

        # 2) 先计算所有 symbol 的 regime（这里简化：单币时先用一次）
        # 在 run_all 中会做完整 weights 计算
        return {
            "regime": regime,
            "df_ltf": df_ltf,
            "df_mtf": df_mtf,
            "df_htf": df_htf,
        }

    def run_all(self, symbol_data: Dict[str, Dict]) -> Dict[str, Dict]:
        # 1) 收集每个 symbol 的 regime
        regimes = {sym: info["regime"] for sym, info in symbol_data.items()}

        # 2) 计算 AI 权重
        weights = self.weight_engine.compute_weights(regimes, self.states)

        results = {}
        for sym, info in symbol_data.items():
            state = self.states[sym]
            regime = info["regime"]
            df_ltf = info["df_ltf"]

            equity_before = state.equity

            equity_after, trades, wins, max_dd = self._apply_trade_logic(
                sym,
                df_ltf=df_ltf,
                regime=regime,
                symbol_weight=weights[sym],
                state=state,
            )

            pnl = equity_after - equity_before
            win_rate = wins / trades * 100 if trades > 0 else 0.0

            results[sym] = {
                "pnl": pnl,
                "trades": trades,
                "wins": wins,
                "win_rate": win_rate,
                "max_dd": max_dd * 100,
                "regime": regime,
                "weight": weights[sym],
            }

        return results


# ========= 数据加载 & 总控 =========

def load_multi_tf_data(
    sym: str,
    days: int,
    data_source: str,
    local_engine: LocalDataEngine,
    real_engine: RealMarketData,
):
    # 低周期：5m，用 days
    if data_source == "local":
        df_ltf = local_engine.load_klines(sym, "5m", days)
        df_mtf = local_engine.load_klines(sym, "1h", days + 3)
        df_htf = local_engine.load_klines(sym, "4h", days + 7)
    else:
        df_ltf = real_engine.get_recent_klines(sym, "5m", days)
        df_mtf = real_engine.get_recent_klines(sym, "1h", days + 3)
        df_htf = real_engine.get_recent_klines(sym, "4h", days + 7)

    logger.info(
        "📥 %s 5m=%d, 1h=%d, 4h=%d (source=%s)",
        sym, len(df_ltf), len(df_mtf), len(df_htf), data_source
    )

    return df_ltf, df_mtf, df_htf


def run_backtest(
    symbols: List[str],
    days: int,
    data_source: str,
    initial_capital: float = 10000.0,
):
    logger.info("🚀 SmartBacktest V14 启动")
    logger.info("🪙 币种: %s", symbols)
    logger.info("📅 回测天数: %d", days)
    logger.info("📊 数据源: %s", data_source)

    local_engine = LocalDataEngine(base_dir="data", exchange="binance")
    real_engine = RealMarketData()

    engine = SmartBacktestV14(symbols=symbols, initial_capital=initial_capital)

    symbol_data: Dict[str, Dict] = {}

    for sym in symbols:
        try:
            df_ltf, df_mtf, df_htf = load_multi_tf_data(
                sym, days, data_source, local_engine, real_engine
            )
            if df_ltf.empty or df_mtf.empty or df_htf.empty:
                logger.warning("⚠️ %s 数据为空，跳过", sym)
                continue

            symbol_data[sym] = engine.run_symbol(sym, df_ltf, df_mtf, df_htf)
        except Exception as e:
            logger.exception("❌ %s 预处理失败: %s", sym, e)

    # 真正运行所有 symbol（包含 AI 权重）
    results = engine.run_all(symbol_data)

    # 汇总
    total_pnl = sum(v["pnl"] for v in results.values())
    total_trades = sum(v["trades"] for v in results.values())
    total_wins = sum(v["wins"] for v in results.values())
    max_dd = min(v["max_dd"] for v in results.values()) if results else 0.0

    total_win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0.0

    print("\n========== 📈 SmartBacktest V14 报告 ==========")
    print(f"总收益: {total_pnl:.2f}")
    print(f"总交易数: {total_trades}")
    print(f"总胜率: {total_win_rate:.2f}%")
    print(f"最大回撤: {max_dd:.2f}%\n")

    print("按币种：")
    for sym, r in results.items():
        regime = r["regime"]
        print(
            f"- {sym}: pnl={r['pnl']:.2f}, trades={r['trades']}, "
            f"win={r['win_rate']:.2f}%, DD={r['max_dd']:.2f}%, "
            f"regime={regime.regime}, trend={regime.trend_score:.2f}, "
            f"range={regime.range_score:.2f}, weight={r['weight']:.2f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC/USDT,ETH/USDT",
        help="逗号分隔的交易对，如: BTC/USDT,ETH/USDT",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="回测天数",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="local",
        choices=["local", "real"],
        help="数据源: local 或 real",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="初始资金",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    run_backtest(
        symbols=symbols,
        days=args.days,
        data_source=args.data_source,
        initial_capital=args.initial_capital,
    )
