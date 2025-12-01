"""
strategy_ab_v20_1.py

V20_1: 策略A（趋势跟踪）+ 策略B（多因子融合）+ 多空 + 杠杆 + ATR 动态止盈止损（修复版）
"""

import argparse
import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from local_data_engine import load_local_kline
except Exception:
    def load_local_kline(*a, **k):
        raise RuntimeError("local_data_engine.load_local_kline 未找到")


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


@dataclass
class BacktestResult:
    symbol: str
    strategy: str
    trades: int
    win_rate: float
    total_return: float
    max_drawdown: float
    sharpe: float


def ensure_ohlc(df):
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="ignore")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp")
    df = df.sort_index()
    return df[["open","high","low","close"]].copy()


def calc_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    prev = close.shift(1)
    tr = pd.concat([high-low, (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_max_drawdown(eq):
    return float((eq/eq.cummax()-1).min())


def calc_sharpe(ret, periods=365*24):
    r = ret.dropna()
    if len(r)<5: return 0.0
    mu, sigma = r.mean(), r.std()
    if sigma==0: return 0.0
    return float((mu*periods)/(sigma*np.sqrt(periods)))


def zscore(s, w=200):
    m = s.rolling(w).mean()
    sd = s.rolling(w).std()
    return (s-m)/(sd+1e-9)


# ============ 策略 A ============

def build_trend_signals(df):
    df = df.copy()
    close = df["close"]

    ema_fast = close.ewm(span=50, adjust=False).mean()
    ema_slow = close.ewm(span=200, adjust=False).mean()

    trend_raw = (ema_fast - ema_slow) / (close+1e-9)
    df["trend_raw"] = trend_raw

    df["signal_dir_A"] = 0
    df.loc[trend_raw>0,"signal_dir_A"] = 1
    df.loc[trend_raw<0,"signal_dir_A"] = -1

    abs_t = trend_raw.abs()
    lo, hi = abs_t.quantile(0.1), abs_t.quantile(0.9)
    conf = (abs_t-lo)/(hi-lo+1e-9)
    df["signal_conf_A"] = conf.clip(0,1)

    return df.dropna().copy()


# ============ 策略 B ============

def build_multifactor_signals(df):
    df = df.copy()
    close = df["close"]

    ema_fast = close.ewm(span=50).mean()
    ema_slow = close.ewm(span=200).mean()
    trend_raw = (ema_fast-ema_slow)/(close+1e-9)

    mom = close/close.shift(24)-1.0
    vol = -close.pct_change().rolling(48).std()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100/(1+(gain/(loss+1e-9)))
    rsi_raw = (rsi-50)/50

    whale = pd.Series(0,index=df.index)
    exflow = pd.Series(0,index=df.index)
    senti = pd.Series(0,index=df.index)

    f = (
        0.35*zscore(trend_raw)+
        0.25*zscore(mom)+
        0.15*zscore(vol)+
        0.15*zscore(rsi_raw)+
        0.05*zscore(whale)+
        0.03*zscore(exflow)+
        0.02*zscore(senti)
    )

    df["factor_score_B_raw"] = f
    df["signal_dir_B"] = np.sign(f)

    abs_f = f.abs()
    lo, hi = abs_f.quantile(0.1), abs_f.quantile(0.9)
    conf = (abs_f-lo)/(hi-lo+1e-9)
    df["signal_conf_B"] = conf.clip(0,1)

    return df.dropna().copy()


# ============ 带杠杆回测 ============

def backtest_with_leverage(df, sig_dir_col, sig_conf_col, capital=10000):
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    atr = calc_atr(df,14)
    df["atr"] = atr

    eq = [capital]
    pos_dir = 0
    pos_lev = 0
    entry_price=0
    entry_i=0
    high_run=0
    low_run=0

    rets=[0]

    trades=[]
    for i in range(1,len(df)):
        pr0 = close.iloc[i-1]
        pr1 = close.iloc[i]

        sig_dir = int(df[sig_dir_col].iloc[i])
        conf = float(df[sig_conf_col].iloc[i])
        lev = 3 + 7*conf

        eq_prev = eq[-1]
        if pos_dir!=0:
            r = pr1/pr0 -1
            eq_now = eq_prev*(1+r*pos_dir*pos_lev)
            rets.append(r*pos_dir*pos_lev)
        else:
            eq_now = eq_prev
            rets.append(0)

        cur_atr = atr.iloc[i]
        if pos_dir!=0:
            high_run=max(high_run,high.iloc[i])
            low_run=min(low_run,low.iloc[i])

        exit_flag=False
        if pos_dir!=0 and not np.isnan(cur_atr):
            if pos_dir>0:
                sl=entry_price-2.5*cur_atr
                trail=high_run-1.5*cur_atr
                if low.iloc[i]<=sl or low.iloc[i]<=trail:
                    exit_flag=True
            else:
                sl=entry_price+2.5*cur_atr
                trail=low_run+1.5*cur_atr
                if high.iloc[i]>=sl or high.iloc[i]>=trail:
                    exit_flag=True

        if pos_dir!=0 and sig_dir!=0 and sig_dir!=pos_dir:
            exit_flag=True

        if exit_flag:
            trades.append((entry_i,i,pos_dir,pos_lev,entry_price,pr1))
            pos_dir=0

        if pos_dir==0 and sig_dir!=0 and not np.isnan(cur_atr):
            pos_dir=sig_dir
            pos_lev=lev
            entry_price=pr1
            entry_i=i
            high_run=high.iloc[i]
            low_run=low.iloc[i]

        eq.append(eq_now)

    eq_series=pd.Series(eq,index=df.index)
    ret_series=pd.Series(rets,index=df.index)

    tot=eq_series.iloc[-1]/capital -1
    dd=calc_max_drawdown(eq_series)
    sharpe=calc_sharpe(ret_series)

    win=0
    for e in trades:
        _,_,d,l,ep,ex=e
        if d>0:
            pnl=(ex/ep -1)*l
        else:
            pnl=(ep/ex -1)*l
        if pnl>0: win+=1

    wc=len(trades)
    wr= win/wc if wc>0 else 0

    return BacktestResult("", "", wc, wr, tot, dd, sharpe), eq_series


# ============ A/B 运行 ============

def run_symbol_A_B(sym, days, interval, capital):
    df_raw = load_local_kline(sym, interval, days)
    df = ensure_ohlc(df_raw)

    dfA = build_trend_signals(df)
    dfA["high"]=df["high"]
    dfA["low"]=df["low"]

    dfB = build_multifactor_signals(df)
    dfB["high"]=df["high"]
    dfB["low"]=df["low"]

    resA,eqA = backtest_with_leverage(dfA,"signal_dir_A","signal_conf_A",capital)
    resA.symbol=sym; resA.strategy="A_trend"

    resB,eqB = backtest_with_leverage(dfB,"signal_dir_B","signal_conf_B",capital)
    resB.symbol=sym; resB.strategy="B_multifactor"

    return resA,resB


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--symbols",type=str,required=True)
    p.add_argument("--days",type=int,default=365)
    p.add_argument("--interval",type=str,default="1h")
    p.add_argument("--capital",type=float,default=10000)
    a=p.parse_args()

    syms=[s.strip() for s in a.symbols.split(",")]

    A=[]; B=[]

    for s in syms:
        try:
            rA,rB=run_symbol_A_B(s,a.days,a.interval,a.capital)
            A.append(rA); B.append(rB)
        except Exception as e:
            logging.exception(f"{s} error: {e}")

    print("\n=== 策略A（趋势） ===")
    for r in A:
        print(f"{r.symbol}: trades={r.trades}, win={r.win_rate:.2f}, "
              f"ret={r.total_return:.4f}, dd={r.max_drawdown:.4f}, sharpe={r.sharpe:.2f}")

    print("\n=== 策略B（多因子） ===")
    for r in B:
        print(f"{r.symbol}: trades={r.trades}, win={r.win_rate:.2f}, "
              f"ret={r.total_return:.4f}, dd={r.max_drawdown:.4f}, sharpe={r.sharpe:.2f}")

if __name__=="__main__": main()
