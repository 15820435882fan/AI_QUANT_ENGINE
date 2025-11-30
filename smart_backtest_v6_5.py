import pandas as pd
import numpy as np
import argparse
import logging
import random
import time


# -------------------------------
# 日志
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - SmartBacktest - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ================================================================================
# 📌 AIEngineV3 —— 五因子智能模型
# ================================================================================
class AIEngineV3:
    """
    核心理念：
    趋势（EMA）+ 动能（MACD）+ 波动率（ATR）+ 统计吸引（VWAP 偏离）+ 情绪（RSI）
    最终给出一个 -1 到 +1 的 AI 总评分。
    """

    def __init__(self):
        # 权重可调，未来可用 AI 自动学习
        self.weights = {
            "trend": 0.28,
            "macd": 0.22,
            "vol": 0.18,
            "vwap": 0.18,
            "rsi": 0.14,
        }

    # -------------------------------
    # 计算指标
    # -------------------------------
    def _indicators(self, df):
        df = df.copy()

        # ✦ EMA 趋势
        df["ema_fast"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()
        df["trend_score"] = np.tanh((df["ema_fast"] - df["ema_slow"]) / df["close"])

        # ✦ MACD
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["signal"] = df["macd"].ewm(span=9).mean()
        df["hist"] = df["macd"] - df["signal"]
        df["macd_score"] = np.tanh(df["hist"] * 20)

        # ✦ 波动率（ATR）
        df["hl"] = df["high"] - df["low"]
        df["hc"] = abs(df["high"] - df["close"].shift())
        df["lc"] = abs(df["low"] - df["close"].shift())
        df["tr"] = df[["hl", "hc", "lc"]].max(axis=1)
        df["atr"] = df["tr"].rolling(14).mean()
        df["atr_norm"] = df["atr"] / df["close"]
        df["vol_score"] = np.tanh((df["atr_norm"] - df["atr_norm"].rolling(30).mean()) * 10)

        # ✦ VWAP 偏离
        df["typ"] = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (df["typ"] * df["volume"]).cumsum() / df["volume"].cumsum()
        df["vwap_score"] = np.tanh((df["close"] - df["vwap"]) / df["close"] * 8)

        # ✦ RSI
        delta = df["close"].diff()
        up = np.maximum(delta, 0)
        down = -np.minimum(delta, 0)
        rs = up.rolling(14).mean() / (down.rolling(14).mean() + 1e-8)
        df["rsi"] = 100 - 100 / (1 + rs)
        df["rsi_score"] = np.tanh((50 - df["rsi"]) / 15)

        return df

    # -------------------------------
    # AI 打分合成
    # -------------------------------
    def _ai_score(self, row):
        score = (
            row["trend_score"] * self.weights["trend"] +
            row["macd_score"] * self.weights["macd"] +
            row["vol_score"] * self.weights["vol"] +
            row["vwap_score"] * self.weights["vwap"] +
            row["rsi_score"] * self.weights["rsi"]
        )
        return float(np.clip(score, -1, 1))

    # -------------------------------
    # 生成信号：1=做多，-1=做空，0=观望
    # -------------------------------
    def generate_signals(self, df):
        df = self._indicators(df)
        df["ai_score"] = df.apply(self._ai_score, axis=1)

        # 自适应阈值：随波动率变化
        df["threshold"] = 0.25 + df["atr_norm"] * 1.2
        df["threshold"] = df["threshold"].clip(0.25, 0.7)

        signals = []
        for _, row in df.iterrows():
            if row["ai_score"] > row["threshold"]:
                signals.append(1)
            elif row["ai_score"] < -row["threshold"]:
                signals.append(-1)
            else:
                signals.append(0)

        return signals


# ================================================================================
# 📌 模拟市场数据
# ================================================================================
def generate_mock_data(days=30, interval=5):
    rows = int(days * 24 * 60 / interval)
    base = 30000
    rng = np.cumsum(np.random.randn(rows) * 50)
    close = base + rng
    high = close + np.random.rand(rows) * 20
    low = close - np.random.rand(rows) * 20
    volume = np.random.rand(rows) * 100

    return pd.DataFrame({
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })


# ================================================================================
# 📌 回测主类
# ================================================================================
class SmartBacktest:
    def __init__(self, symbols, days, engine):
        self.symbols = symbols
        self.days = days
        self.engine = engine
        self.balance = 10000
        self.safebox = 0

    def _run_symbol(self, sym):
        logging.info(f"🔍 测试币种: {sym}")
        df = generate_mock_data(self.days)
        logging.info(f"📊 使用模拟市场数据: {sym}")

        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        signals = self.engine.generate_signals(df)

        last_sig = 0
        entry = 0

        for price, sig in zip(df["close"], signals):
            if sig != 0 and last_sig == 0:
                entry = price
                last_sig = sig

            elif sig == 0 and last_sig != 0:
                pnl = (price - entry) * last_sig
                self.balance += pnl / 10
                last_sig = 0

    def run(self):
        logging.info("🚀 开始回测 ...")

        for s in self.symbols:
            self._run_symbol(s)

        logging.info("=" * 80)
        logging.info("🧠 智能量化交易系统 - 回测报告")
        logging.info("=" * 80)
        logging.info(f"最终余额: {self.balance:.2f}, 保险柜: {self.safebox:.2f}")


# ================================================================================
# 📌 主程序入口
# ================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT,SOL/USDT")
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()

    symbols = args.symbols.split(",")

    engine = AIEngineV3()
    bt = SmartBacktest(symbols, args.days, engine)
    bt.run()


if __name__ == "__main__":
    main()
