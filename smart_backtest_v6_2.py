import argparse
import logging
import numpy as np
import pandas as pd
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional


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

        sigs = []
        for i in range(len(close)):
            if i < self.slow:
                sigs.append(AISignal("hold", 0))
                continue

            if fast_ma[i] > slow_ma[i]:
                sigs.append(AISignal("long", 0.55))
            elif fast_ma[i] < slow_ma[i]:
                sigs.append(AISignal("short", 0.55))
            else:
                sigs.append(AISignal("hold", 0.2))

        return sigs


# ============================================================
#  AI Engine v6.2 — fixed, lightweight, high-signal-rate
# ============================================================

class AIProdEngine:
    def __init__(self, fast=12, slow=40):
        self.base = BaselineEngine(fast, slow)

    def generate_signals(self, df: pd.DataFrame) -> List[AISignal]:
        close = df["close"]
        baseline = self.base.generate_signals(df)

        # Pre-calc lightweight indicators
        rsi14 = rsi(close, 14)
        vola = recent_volatility(close, 30)
        trend30 = close.pct_change(30).fillna(0)

        signals = []
        for i in range(len(close)):
            if i < 40:
                signals.append(AISignal("hold", 0))
                continue

            b = baseline[i]

            # Trend score (lightweight)
            ts = np.tanh(trend30[i] * 5)

            # Vola score (prefer mid-volatility)
            vs = np.exp(-abs(vola[i] - vola.mean()) * 20) if not np.isnan(vola[i]) else 0.3

            # RSI modifier
            ri = rsi14[i]
            if np.isnan(ri):
                rsi_score = 0.2
            else:
                rsi_score = 1 - abs(ri - 50) / 50  # middle is better

            # Final confidence
            conf = (ts * 0.4 + vs * 0.3 + rsi_score * 0.3)
            conf = float(max(0, min(1, conf)))

            # Action selection
            if conf < 0.05:
                signals.append(AISignal("hold", conf))
                continue

            # final direction = baseline direction
            signals.append(AISignal(b.action, conf))

        return signals


# ============================================================
#  Synthetic Market for Simulation
# ============================================================

def generate_synthetic_klines(n=8640, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    prices = [100]
    for _ in range(n):
        # Realistic clustered volatility
        drift = np.random.normal(0, 0.001)
        shock = np.random.normal(0, 0.002 if random.random() < 0.1 else 0.0005)
        prices.append(prices[-1] * (1 + drift + shock))

    df = pd.DataFrame({
        "open": prices[:-1],
        "high": [p * 1.002 for p in prices[:-1]],
        "low": [p * 0.998 for p in prices[:-1]],
        "close": prices[1:],
        "volume": np.random.randint(100, 500, size=n)
    })
    return df


# ============================================================
#  Account & Risk Control
# ============================================================

class Account:
    def __init__(self, initial=10000, leverage=3.0):
        self.balance = initial
        self.leverage = leverage
        self.position = 0
        self.entry = 0
        self.insurance = 0
        self.pnl_history = []
        self.max_equity = initial

        # Risk control
        self.consec_loss = 0
        self.cooldown = 0

    def update(self, price: float):
        if self.position != 0:
            pnl = (price - self.entry) * self.position
            equity = self.balance + pnl
        else:
            equity = self.balance

        self.max_equity = max(self.max_equity, equity)
        dd = (equity - self.max_equity) / self.max_equity
        return equity, dd

    def open(self, direction: str, price: float, conf: float):
        if self.cooldown > 0:
            return False

        risk_size = self.balance * 0.1 * conf
        qty = risk_size * self.leverage / price
        self.position = qty if direction == "long" else -qty
        self.entry = price
        return True

    def close(self, price: float):
        if self.position == 0:
            return 0

        pnl = (price - self.entry) * self.position
        self.balance += pnl
        self.pnl_history.append(pnl)

        #抽佣：大于 3% 即入保险柜
        if pnl > self.balance * 0.03:
            take = pnl * 0.25
            self.insurance += take
            self.balance -= take

        if pnl < 0:
            self.consec_loss += 1
        else:
            self.consec_loss = 0

        self.position = 0
        return pnl


# ============================================================
#  Backtest System
# ============================================================

class SmartBacktest:
    def __init__(self, symbols, days, engine_name, use_real):
        self.symbols = symbols
        self.days = days
        self.use_real = use_real
        self.engine = AIProdEngine() if engine_name == "ai_prod" else BaselineEngine()
        self.account = Account()

    def load_data(self, sym):
        logger.info(f"📊 使用模拟市场数据: {sym}")
        return generate_synthetic_klines(self.days * 288)

    def run_symbol(self, sym):
        df = self.load_data(sym)
        sigs = self.engine.generate_signals(df)

        for i in range(len(df)):
            price = df["close"][i]
            sig = sigs[i]

            eq, dd = self.account.update(price)
            if dd < -0.6:
                break

            if self.account.cooldown > 0:
                self.account.cooldown -= 1
                continue

            if self.account.position == 0:
                if sig.action in ["long", "short"] and sig.confidence > 0.1:
                    self.account.open(sig.action, price, sig.confidence)
            else:
                if sig.action == "hold":
                    self.account.close(price)

            if self.account.consec_loss >= 8:
                self.account.cooldown = 288
                self.account.consec_loss = 0

    def run(self):
        for s in self.symbols:
            logger.info(f"🔍 测试币种: {s}")
            self.run_symbol(s)

        self.report()

    def report(self):
        total_trades = len(self.account.pnl_history)
        total_pnl = sum(self.account.pnl_history)

        logger.info("=" * 80)
        logger.info("🧠 智能量化交易系统 - 回测报告")
        logger.info("=" * 80)
        logger.info(f"总交易: {total_trades}, 总收益: {total_pnl:.2f}")
        logger.info(f"最终余额: {self.account.balance:.2f}, 保险柜: {self.account.insurance:.2f}")


# ============================================================
#  CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTC/USDT")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--engine", type=str, default="ai_prod")
    parser.add_argument("--use-real-data", action="store_true")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    bt = SmartBacktest(symbols, args.days, args.engine, args.use_real_data)
    bt.run()


if __name__ == "__main__":
    main()
