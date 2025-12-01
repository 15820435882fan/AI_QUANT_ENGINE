# smart_backtest_v14_2.py
# V14_2: 在 V14 基础上修复 Regime 评分过大、权重失效、冷静期过于频繁的问题

import argparse
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from local_data_engine import LocalDataEngine
from real_market_data_v3 import RealMarketData  # 如果你叫 v2，这里改一下即可

logger = logging.getLogger("SmartBacktestV14_2")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ================ 工具函数 =================

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


def ma_slope(series: pd.Series, window: int = 10) -> float:
    if series.isna().sum() > 0 or len(series) < window + 1:
        return 0.0
    # 简单差分斜率
    return float((series.iloc[-1] - series.iloc[-window - 1]) / window / (series.iloc[-window - 1] + 1e-9))


# ================ Regime & 状态结构 =================

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
    AI 权重引擎：
    - 输入 Regime + 历史 pnl，输出一个 0~1 权重
    """

    def __init__(self, symbols: List[str]):
        self.symbols = symbols

    def score_symbol(self, sym: str, regime: RegimeInfo, state: SymbolState) -> float:
        # 趋势越强、震荡越弱 → 越好
        trend_part = regime.trend_score
        range_penalty = 1.0 - regime.range_score

        # ETH 天然更震荡，趋势分稍微打折
        if sym.upper().startswith("ETH"):
            trend_part *= 0.8

        # 近期 pnl 趋势（最后 20 笔）
        if state.pnl_history:
            recent = state.pnl_history[-20:]
            pnl_avg = float(np.mean(recent))
            pnl_norm = np.tanh(pnl_avg / (state.equity + 1e-9))  # -1~1
        else:
            pnl_norm = 0.0

        # 综合评分：0~1
        raw = 0.55 * trend_part + 0.25 * range_penalty + 0.20 * (pnl_norm + 1) / 2
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
                scores[sym] = self.score_symbol(sym, r, s)

        total = sum(scores.values())
        if total <= 1e-9:
            # 所有都很差 → 均分一点权重
            n = len(self.symbols)
            return {sym: 1.0 / n for sym in self.symbols}

        return {sym: v / total for sym, v in scores.items()}


# ================ 核心回测引擎 =================

class SmartBacktestV14_2:
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
            sym: SymbolState(
                name=sym,
                equity=initial_capital / len(symbols),
                max_equity=initial_capital / len(symbols),
            )
            for sym in symbols
        }

        self.weight_engine = AIWeightEngine(symbols)

    # ------ Regime 检测（修复版） ------

    def _detect_regime(
        self,
        sym: str,
        df_ltf: pd.DataFrame,
        df_mtf: pd.DataFrame,
        df_htf: pd.DataFrame,
    ) -> RegimeInfo:
        """
        用多周期 MA 斜率 + BOLL + ATR 做 Regime 判别
        修复 V14 中趋势评分几乎总为 1 的问题
        """

        # 取尾部窗口
        mtf_tail = df_mtf.tail(200)
        htf_tail = df_htf.tail(200)
        ltf_tail = df_ltf.tail(300)

        # --- 趋势部分：看 1h & 4h MA 斜率 ---
        ma_mtf_fast = compute_ma(mtf_tail, 34)
        ma_mtf_slow = compute_ma(mtf_tail, 89)
        ma_htf_fast = compute_ma(htf_tail, 21)
        ma_htf_slow = compute_ma(htf_tail, 55)

        mtf_trend_line = (ma_mtf_fast + ma_mtf_slow) / 2
        htf_trend_line = (ma_htf_fast + ma_htf_slow) / 2

        mtf_s = ma_slope(mtf_trend_line.dropna(), window=24)  # 约 1 天
        htf_s = ma_slope(htf_trend_line.dropna(), window=10)  # 约 2 天

        # 经验缩放：一般斜率在 10^-4 ~ 10^-3 级别
        trend_raw = abs(mtf_s) * 300 + abs(htf_s) * 500
        trend_score = max(0.0, min(1.0, trend_raw))

        # ETH 趋势再打 0.85 折，防止过度乐观
        if sym.upper().startswith("ETH"):
            trend_score *= 0.85

        # --- 震荡部分：BOLL 宽度 + ATR 压缩 ---
        close_ltf = ltf_tail["close"]
        mid = close_ltf.rolling(20).mean()
        std = close_ltf.rolling(20).std()
        upper = mid + std
        lower = mid - std

        bw = ((upper - lower) / (close_ltf + 1e-9)).iloc[-1]  # 相对带宽
        atr_ltf = compute_atr(ltf_tail, 14)
        atr_ratio = (atr_ltf / (close_ltf + 1e-9)).iloc[-1]

        # 适度放大，不再用 tanh*200 那种夸张缩放
        bw_norm = max(0.0, min(1.0, bw * 40))       # 带宽越大 → 趋势/波动越强
        atr_norm = max(0.0, min(1.0, atr_ratio * 120))

        # 震荡：带宽中等 + ATR 不高时更震荡
        range_score = (bw_norm * 0.5 + (1 - atr_norm) * 0.5)
        range_score = max(0.0, min(1.0, range_score))

        # 波动质量：太低 or 太高都不好
        vol_score = 1.0 - abs(atr_norm - 0.3)
        vol_score = max(0.0, min(1.0, vol_score))

        # --- Regime 决策（修正版阈值） ---
        if trend_score > 0.55 and range_score < 0.45:
            regime = "trend"
        elif range_score > 0.65 and trend_score < 0.5:
            regime = "range"
        else:
            regime = "mixed"

        return RegimeInfo(
            trend_score=float(trend_score),
            range_score=float(range_score),
            vol_score=float(vol_score),
            regime=regime,
        )

    # ------ 单根 K 线方向信号（简单多头） ------

    def _generate_signal_row(
        self,
        row_ltf: pd.Series,
        trend_regime: RegimeInfo,
    ) -> int:
        """
        方向决策：
        - 只在 regime 为 trend 或 mixed 且 trend_score 足够高时开单
        - 简单用 MA 快慢线 + close 相对 MA 位置 做方向
        return: 1=多头, -1=空头, 0=观望
        """
        if trend_regime.regime == "range" and trend_regime.trend_score < 0.5:
            return 0

        if trend_regime.trend_score < 0.4:
            return 0

        close = row_ltf["close"]
        ma_fast = row_ltf.get("ma_fast", np.nan)
        ma_slow = row_ltf.get("ma_slow", np.nan)

        if np.isnan(ma_fast) or np.isnan(ma_slow):
            return 0

        if ma_fast > ma_slow and close > ma_fast:
            return 1
        return 0

    # ------ 核心交易循环（与 V13/V14 基本一致） ------

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
        - 止盈：ATR * 2.5（大约 RR=2）
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
        df["ma_fast"] = compute_ma(df, 20)
        df["ma_slow"] = compute_ma(df, 60)

        # 仓位因子：Regime 越偏趋势 + 波动质量越好 → 仓位越高
        regime_factor = 0.5 * regime.trend_score + 0.5 * regime.vol_score
        regime_factor = 0.3 + 0.7 * regime_factor   # 最低 0.3，最高 1.0

        # 最终仓位比例（相对这个 symbol 的 equity）
        position_scale = symbol_weight * regime_factor  # 0~1 左右

        for i, (idx, row) in enumerate(df.iterrows()):
            price = row["close"]
            this_atr = atr.iloc[i]

            # 冷静期检查：冷静期内不再开新仓
            if cooldown_until >= 0 and i < cooldown_until:
                # 有仓位则继续管理（这里我们只有平仓，不再加仓）
                if position > 0 and this_atr > 0:
                    stop_loss = entry_price - this_atr * 1.2
                    take_profit = entry_price + this_atr * 2.5

                    if price <= stop_loss or price >= take_profit:
                        pnl = (price - entry_price) * position
                        equity += pnl
                        trades += 1
                        if pnl >= 0:
                            wins += 1
                            cold_streak = 0
                        else:
                            losses += 1
                            # 冷静期中再亏，不再叠加冷静期长度，只更新 streak
                            cold_streak += 1
                        state.pnl_history.append(pnl)
                        position = 0
                        entry_price = 0

                # 更新回撤
                if equity > max_equity:
                    max_equity = equity
                dd = (equity - max_equity) / max_equity
                if dd < max_dd:
                    max_dd = dd
                continue

            # 不在冷静期：允许生成新信号
            direction = self._generate_signal_row(row, regime)

            # 开仓
            if position == 0 and direction == 1 and this_atr > 0:
                risk_per_trade = equity * 0.01  # 每笔 risk ~1%
                qty = (risk_per_trade / (this_atr * 1.2)) * position_scale
                if qty > 0:
                    position = qty
                    entry_price = price
                    continue

            # 管理持仓
            if position > 0 and this_atr > 0:
                stop_loss = entry_price - this_atr * 1.2
                take_profit = entry_price + this_atr * 2.5

                closed = False
                if price <= stop_loss:
                    pnl = (price - entry_price) * position
                    equity += pnl
                    trades += 1
                    losses += 1
                    cold_streak += 1
                    state.pnl_history.append(pnl)
                    closed = True

                elif price >= take_profit:
                    pnl = (price - entry_price) * position
                    equity += pnl
                    trades += 1
                    wins += 1
                    cold_streak = 0
                    state.pnl_history.append(pnl)
                    closed = True

                if closed:
                    position = 0
                    entry_price = 0

                    # 冷静期触发逻辑（修复：进入冷静期时重置 cold_streak）
                    if cold_streak >= self.loss_cooldown_n:
                        # 冷静期长度与 Regime 和 range_score 相关
                        base_cool = 60  # 60 根 5m K → 5 小时
                        extra = int(60 * regime.range_score)
                        cool_len = base_cool + extra
                        cooldown_until = i + cool_len
                        logger.info(
                            "🧊 %s 连续亏损(%d) → 冷静期 %d bars",
                            sym, cold_streak, cool_len,
                        )
                        cold_streak = 0  # ⭐ 关键：进入冷静期后 streak 归零

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
        regime = self._detect_regime(sym, df_ltf, df_mtf, df_htf)
        return {
            "regime": regime,
            "df_ltf": df_ltf,
            "df_mtf": df_mtf,
            "df_htf": df_htf,
        }

    def run_all(self, symbol_data: Dict[str, Dict]) -> Dict[str, Dict]:
        regimes = {sym: info["regime"] for sym, info in symbol_data.items()}
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


# ================ 数据加载 & 总控 =================

def load_multi_tf_data(
    sym: str,
    days: int,
    data_source: str,
    local_engine: LocalDataEngine,
    real_engine: RealMarketData,
):
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
    logger.info("🚀 SmartBacktest V14_2 启动")
    logger.info("🪙 币种: %s", symbols)
    logger.info("📅 回测天数: %d", days)
    logger.info("📊 数据源: %s", data_source)

    local_engine = LocalDataEngine(base_dir="data", exchange="binance")
    real_engine = RealMarketData()

    engine = SmartBacktestV14_2(symbols=symbols, initial_capital=initial_capital)

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

    results = engine.run_all(symbol_data)

    total_pnl = sum(v["pnl"] for v in results.values())
    total_trades = sum(v["trades"] for v in results.values())
    total_wins = sum(v["wins"] for v in results.values())
    max_dd = min(v["max_dd"] for v in results.values()) if results else 0.0

    total_win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0.0

    print("\n========== 📈 SmartBacktest V14_2 报告 ==========")
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
