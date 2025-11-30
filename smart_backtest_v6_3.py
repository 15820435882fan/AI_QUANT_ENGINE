import argparse
import logging
import numpy as np
import pandas as pd
import random
from dataclasses import dataclass
from typing import List


# ============================================================
#  Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - SmartBacktest - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SmartBacktest")


# ============================================================
#  Utility: Lightweight indicators (fast!)
# ============================================================

def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window).mean()
    ma_down = down.rolling(window).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def recent_volatility(series: pd.Series, window=30):
    return series.pct_change().rolling(window).std()


# ============================================================
#  AI Signal Structure
# ============================================================

@dataclass
class AISignal:
    action: str         # 'long', 'short', or 'hold'
    confidence: float   # 0~1


# ============================================================
#  Baseline Engine (fixed and stable)
# ============================================================

class BaselineEngine:
    def __init__(self, fast=12, slow=40):
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> List[AISignal]:
        close = df["close"]
        fast_ma = ema(close, self.fast)
        slow_ma = ema(close, self.slow)

        sigs: List[AISignal] = []
        for i in range(len(close)):
            if i < self.slow:
                sigs.append(AISignal("hold", 0.0))
                continue

            if fast_ma.iloc[i] > slow_ma.iloc[i]:
                sigs.append(AISignal("long", 0.6))
            elif fast_ma.iloc[i] < slow_ma.iloc[i]:
                sigs.append(AISignal("short", 0.6))
            else:
                sigs.append(AISignal("hold", 0.2))

        return sigs


# ============================================================
#  AI Engine v6.3 — lightweight, higher signal rate
# ============================================================

class AIProdEngine:
    def __init__(self, fast=12, slow=40):
        self.base = BaselineEngine(fast, slow)

    def generate_signals(self, df: pd.DataFrame) -> List[AISignal]:
        close = df["close"]
        baseline = self.base.generate_signals(df)

        # Pre-calc indicators
        rsi14 = rsi(close, 14)
        vola = recent_volatility(close, 30)
        trend30 = close.pct_change(30).fillna(0)

        sigs: List[AISignal] = []
        for i in range(len(close)):
            if i < 40:
                sigs.append(AISignal("hold", 0.0))
                continue

            b = baseline[i]

            # Trend score
            ts = float(np.tanh(trend30.iloc[i] * 8))  # stronger scaling

            # Volatility preference (mid > too low/high)
            if not np.isnan(vola.iloc[i]):
                vol_norm = abs(vola.iloc[i] - vola.mean()) / (vola.std() + 1e-9)
                vs = float(np.exp(-vol_norm ** 2))
            else:
                vs = 0.5

            # RSI score (50附近最好)
            ri = rsi14.iloc[i]
            if np.isnan(ri):
                rsi_score = 0.4
            else:
                rsi_score = 1 - abs(ri - 50) / 60  # 允许偏离一点
                rsi_score = max(0.0, min(1.0, rsi_score))

            raw_conf = 0.4 * (ts + 1) / 2 + 0.3 * vs + 0.3 * rsi_score
            conf = float(max(0.05, min(0.95, raw_conf)))

            # 动作沿用 baseline 方向
            if b.action == "hold" and conf < 0.25:
                sigs.append(AISignal("hold", conf))
            else:
                sigs.append(AISignal(b.action, conf))

        return sigs


# ============================================================
#  Synthetic Market for Simulation
# ============================================================

def generate_synthetic_klines(n=8640, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    prices = [100.0]
    for _ in range(n):
        drift = np.random.normal(0, 0.0008)
        shock = np.random.normal(0, 0.0025 if random.random() < 0.08 else 0.0008)
        prices.append(prices[-1] * (1 + drift + shock))

    prices = np.array(prices)
    open_ = prices[:-1]
    close = prices[1:]
    high = np.maximum(open_, close) * (1 + np.random.uniform(0, 0.002, size=n))
    low = np.minimum(open_, close) * (1 - np.random.uniform(0, 0.002, size=n))
    vol = np.random.randint(100, 1000, size=n)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    })
    return df


# ============================================================
#  Account & Risk Control
# ============================================================

class Account:
    def __init__(self, initial=10000.0, leverage=3.0):
        self.initial = initial
        self.balance = initial
        self.leverage = leverage
        self.position = 0.0      # 正多负空
        self.entry = 0.0
        self.insurance = 0.0
        self.pnl_history: List[float] = []
        self.equity_curve: List[float] = []

        # risk state
        self.consec_loss = 0
        self.cooldown = 0
        self.max_equity = initial

    def current_equity(self, price: float) -> float:
        if self.position != 0.0:
            pnl = (price - self.entry) * self.position
            return self.balance + pnl
        return self.balance

    def update_equity(self, price: float) -> float:
        eq = self.current_equity(price)
        self.equity_curve.append(eq)
        if eq > self.max_equity:
            self.max_equity = eq
        return eq

    def open_position(self, direction: str, price: float, conf: float) -> bool:
        if self.cooldown > 0 or self.position != 0.0:
            return False

        # 风险头寸 = 账户余额 * 0.08 * 信心
        risk_size = self.balance * 0.08 * conf
        if risk_size <= 0:
            return False

        qty = risk_size * self.leverage / price
        if qty <= 0:
            return False

        self.position = qty if direction == "long" else -qty
        self.entry = price
        return True

    def close_position(self, price: float):
        if self.position == 0.0:
            return 0.0, 0.0

        pnl = (price - self.entry) * self.position
        self.balance += pnl
        self.pnl_history.append(pnl)

        insurance_take = 0.0
        # 单笔盈利 > 3% 触发抽佣
        if pnl > 0:
            self.consec_loss = 0
            if pnl > self.balance * 0.03:
                insurance_take = pnl * 0.25
                self.insurance += insurance_take
                self.balance -= insurance_take
        else:
            self.consec_loss += 1

        self.position = 0.0
        self.entry = 0.0
        return pnl, insurance_take


# ============================================================
#  Backtest System
# ============================================================

class SmartBacktest:
    def __init__(self, symbols, days, engine_name, use_real_data=False):
        self.symbols = symbols
        self.days = days
        self.use_real_data = use_real_data

        if engine_name == "ai_prod":
            self.engine = AIProdEngine()
        else:
            self.engine = BaselineEngine()

        self.account = Account()
        self.symbol_stats = {
            sym: {"trades": 0, "wins": 0, "pnl": 0.0, "insurance": 0.0}
            for sym in symbols
        }

    def load_data(self, symbol: str) -> pd.DataFrame:
        # 目前统一用模拟数据
        logger.info(f"📊 使用模拟市场数据: {symbol}")
        bars = self.days * 288  # 5m * 288 ≈ 1 天
        return generate_synthetic_klines(n=bars, seed=hash(symbol) % 10_000)

    def _compute_max_drawdown(self) -> float:
        if not self.account.equity_curve:
            return 0.0
        max_eq = self.account.equity_curve[0]
        max_dd = 0.0
        for eq in self.account.equity_curve:
            if eq > max_eq:
                max_eq = eq
            dd = (eq - max_eq) / max_eq
            if dd < max_dd:
                max_dd = dd
        return max_dd  # negative

    def run_single_symbol(self, symbol: str):
        df = self.load_data(symbol)
        signals = self.engine.generate_signals(df)

        stats = self.symbol_stats[symbol]

        for i in range(len(df)):
            price = float(df["close"].iloc[i])
            sig = signals[i]

            # 更新净值曲线
            self.account.update_equity(price)

            # 暂停期
            if self.account.cooldown > 0:
                self.account.cooldown -= 1
                continue

            # 有持仓 -> 检查止损 / 止盈 / 反向
            if self.account.position != 0.0:
                direction = "long" if self.account.position > 0 else "short"
                pnl_now = (price - self.account.entry) * self.account.position
                ret_ratio = pnl_now / max(self.account.balance, 1e-9)

                close_flag = False
                if ret_ratio <= -0.08 or ret_ratio >= 0.10:
                    close_flag = True
                elif sig.action != "hold" and sig.action != direction and sig.confidence > 0.4:
                    close_flag = True

                if close_flag:
                    pnl, ins = self.account.close_position(price)
                    if pnl != 0.0:
                        stats["trades"] += 1
                        stats["pnl"] += pnl
                        stats["insurance"] += ins
                        if pnl > 0:
                            stats["wins"] += 1

                    if self.account.consec_loss >= 6:
                        self.account.cooldown = 288  # 一天
                        self.account.consec_loss = 0

            # 无持仓 -> 根据信号开仓
            if self.account.position == 0.0:
                if sig.action in ("long", "short") and sig.confidence > 0.25:
                    self.account.open_position(sig.action, price, sig.confidence)

        # 结束强制平仓
        if self.account.position != 0.0:
            last_price = float(df["close"].iloc[-1])
            pnl, ins = self.account.close_position(last_price)
            if pnl != 0.0:
                stats["trades"] += 1
                stats["pnl"] += pnl
                stats["insurance"] += ins
                if pnl > 0:
                    stats["wins"] += 1

    def run(self):
        logger.info("🚀 开始回测 ...")
        for sym in self.symbols:
            logger.info(f"🔍 测试币种: {sym}")
            self.run_single_symbol(sym)
        self._report()

    def _report(self):
        total_trades = sum(s["trades"] for s in self.symbol_stats.values())
        total_pnl = sum(s["pnl"] for s in self.symbol_stats.values())
        total_wins = sum(s["wins"] for s in self.symbol_stats.values())

        final_equity = self.account.balance + self.account.insurance
        total_return = (final_equity - self.account.initial) / self.account.initial if self.account.initial > 0 else 0.0

        win_rate = total_wins / total_trades if total_trades > 0 else 0.0
        max_dd = self._compute_max_drawdown()  # negative
        max_dd_pct = -max_dd * 100

        days = self.days if self.days > 0 else 1
        months = max(days / 30.0, 1e-6)
        try:
            monthly_return = (1 + total_return) ** (1 / months) - 1
        except Exception:
            monthly_return = 0.0

        logger.info("=" * 80)
        logger.info("🧠 智能量化交易系统 - 回测报告")
        logger.info("=" * 80)
        logger.info(f"测试币种: {len(self.symbol_stats)} 个")
        logger.info(f"总交易次数: {total_trades} 笔")
        logger.info(f"总收益: ${total_pnl:.2f}")
        logger.info(f"最终资金: ${final_equity:.2f} (账户: {self.account.balance:.2f} + 保险柜: {self.account.insurance:.2f})")
        logger.info(f"平均胜率: {win_rate * 100:.1f}%")
        logger.info(f"最大回撤: {max_dd_pct:.1f}%")
        logger.info(f"粗略年化/月化估算: 月化≈{monthly_return:.1%} （目标≥20%）")
        logger.info("")
        logger.info("📊 各币种表现:")
        for sym, s in self.symbol_stats.items():
            wr = (s["wins"] / s["trades"] * 100) if s["trades"] > 0 else 0.0
            logger.info(
                f"  🟡 {sym}: {s['trades']} 笔, 胜率: {wr:.1f}%, 收益: ${s['pnl']:.2f}, 抽取到保险柜: ${s['insurance']:.2f}"
            )

        logger.info("")
        logger.info("🏦 利润抽取 + 复利模拟（简化版）:")
        logger.info(
            f"  回测结束时账户资金≈${self.account.balance:.2f}，保险柜安全利润≈${self.account.insurance:.2f}，合计总资产≈${final_equity:.2f}。"
        )

        # AI 风险收益评分
        score = self._ai_score(total_return, max_dd, win_rate, total_trades)
        level, comment = self._score_comment(score)

        logger.info("")
        logger.info("🤖 AI 风险收益评分:")
        logger.info(f"  综合得分: {score:.1f} / 100, 等级: {level}, 评语: {comment}")
        logger.info("")
        logger.info("🎉 智能回测完成！")
        logger.info("=" * 80)

    @staticmethod
    def _ai_score(total_return: float, max_dd: float, win_rate: float, trades: int) -> float:
        # 1) 收益因子（最多 40 分）
        ret = max(-1.0, min(3.0, total_return))  # -100% ~ +300%
        ret_score = (ret + 1.0) / 4.0 * 40.0     # -1 -> 0, +3 -> 40

        # 2) 回撤因子（最多 30 分，回撤越小越好）
        dd = min(1.0, max(0.0, -max_dd))        # 0 ~ 1
        dd_score = (1.0 - dd) * 30.0            # 0 回撤 -> 30 分

        # 3) 胜率因子（最多 20 分）
        wr_score = max(0.0, min(1.0, win_rate)) * 20.0

        # 4) 交易样本因子（最多 10 分）
        t = min(1.0, trades / 500.0)
        trade_score = t * 10.0

        score = ret_score + dd_score + wr_score + trade_score
        return max(0.0, min(100.0, score))

    @staticmethod
    def _score_comment(score: float):
        if score >= 80:
            return "A", "收益与风险控制优秀，可以考虑小规模实盘验证。"
        elif score >= 65:
            return "B", "策略表现较好，但仍有回撤或稳定性问题，适合模拟盘长时间观察。"
        elif score >= 50:
            return "C", "策略风险收益比一般，建议先小仓位或仅用作研究参考。"
        elif score >= 35:
            return "D", "风险较大或表现不稳定，不建议用于真实资金。"
        else:
            return "E", "当前策略不建议用于真实资金，可用于反向情绪或继续调参。"


# ============================================================
#  CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTC/USDT")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--engine", type=str, default="ai_prod", choices=["ai_prod", "baseline"])
    parser.add_argument("--use-real-data", action="store_true")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    bt = SmartBacktest(symbols, args.days, args.engine, args.use_real_data)
    bt.run()


if __name__ == "__main__":
    main()
