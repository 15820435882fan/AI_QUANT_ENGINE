# v30_eval_in_env.py
#
# 使用 V30 Transformer 模型 + 老师同一套特征 / 数据，
# 在原始 K 线数据上跑一个简易的回测，检验策略行为和收益表现。
#
# 注意：
#   这里没有直接依赖 TradingEnvV22MultiTFV17，
#   而是用和 V30 数据集相同的特征 & 老师规则，自带一个轻量级回测引擎。
#
# 命令示例：
#   python v30_eval_in_env.py --symbol BTCUSDT --timeframe 1h --days 2920 \
#       --model-path models_v30/v30_transformer_BTCUSDT_1h_best.pt

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from v30_teacher_strategies import teacher_trend_follow, TeacherConfig
from v30_dataset_builder import load_ohlc, build_features
from v30_policy_wrapper import V30TransformerPolicy


@dataclass
class BacktestResult:
    equity_curve: np.ndarray
    timestamps: np.ndarray
    trades: List[Dict]
    final_equity: float
    max_drawdown: float
    win_rate: float
    num_trades: int


def run_backtest_with_actions(
    prices: np.ndarray,
    timestamps: np.ndarray,
    actions: np.ndarray,
    initial_equity: float = 10000.0,
) -> BacktestResult:
    """
    使用给定的动作序列做一个非常简化的多空回测：
      - 0: HOLD
      - 1: 开/持多
      - 2: 开/持空
      - 3: 平仓 -> 变为空仓
    """
    n = len(prices)
    assert len(actions) == n

    equity_curve = np.zeros(n, dtype=np.float64)
    trades: List[Dict] = []

    equity = float(initial_equity)
    entry_equity = equity
    pos_side = 0   # 0: flat, 1: long, -1: short
    entry_price = 0.0
    units = 0.0
    open_trade: Dict | None = None

    for i in range(n):
        price = float(prices[i])
        ts = timestamps[i]

        # 先按当前持仓做一次 mark-to-market
        if pos_side == 1:
            # 多头：市值 = units * price
            equity = units * price
        elif pos_side == -1:
            # 空头：以开仓时 equity 为基准
            equity = entry_equity + (entry_price - price) * units
        # flat 时 equity 不变

        act = int(actions[i])

        if pos_side == 0:
            # 空仓时，只有遇到开仓动作才操作
            if act == 1:  # 开多
                pos_side = 1
                entry_price = price
                entry_equity = equity
                units = equity / price  # 全仓
                open_trade = {
                    "side": "long",
                    "entry_idx": i,
                    "entry_time": ts,
                    "entry_price": price,
                }
            elif act == 2:  # 开空
                pos_side = -1
                entry_price = price
                entry_equity = equity
                units = equity / price
                open_trade = {
                    "side": "short",
                    "entry_idx": i,
                    "entry_time": ts,
                    "entry_price": price,
                }
        else:
            # 持仓中，只有遇到平仓动作才平掉
            if act == 3:
                # 再 mark 一次，获得最终 equity
                if pos_side == 1:
                    equity = units * price
                    pnl = equity - entry_equity
                else:
                    equity = entry_equity + (entry_price - price) * units
                    pnl = equity - entry_equity

                if open_trade is not None:
                    open_trade["exit_idx"] = i
                    open_trade["exit_time"] = ts
                    open_trade["exit_price"] = price
                    open_trade["pnl"] = pnl
                    open_trade["return"] = pnl / max(entry_equity, 1e-9)
                    trades.append(open_trade)
                    open_trade = None

                pos_side = 0
                entry_price = 0.0
                units = 0.0
                entry_equity = equity

        equity_curve[i] = equity

    # 如果最后还在持仓，可以按最后一个价格强制平仓
    if pos_side != 0 and open_trade is not None:
        price = float(prices[-1])
        ts = timestamps[-1]
        if pos_side == 1:
            equity = units * price
            pnl = equity - entry_equity
        else:
            equity = entry_equity + (entry_price - price) * units
            pnl = equity - entry_equity

        open_trade["exit_idx"] = n - 1
        open_trade["exit_time"] = ts
        open_trade["exit_price"] = price
        open_trade["pnl"] = pnl
        open_trade["return"] = pnl / max(entry_equity, 1e-9)
        trades.append(open_trade)

        pos_side = 0
        units = 0.0
        entry_price = 0.0
        entry_equity = equity

        equity_curve[-1] = equity

    # 计算最大回撤
    peak = -1e18
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / max(peak, 1e-9)
        if dd > max_dd:
            max_dd = dd

    # 胜率
    if len(trades) > 0:
        wins = sum(1 for t in trades if t.get("pnl", 0.0) > 0)
        win_rate = wins / len(trades)
    else:
        win_rate = 0.0

    return BacktestResult(
        equity_curve=equity_curve,
        timestamps=timestamps,
        trades=trades,
        final_equity=float(equity_curve[-1]),
        max_drawdown=float(max_dd),
        win_rate=float(win_rate),
        num_trades=len(trades),
    )


def run_v30_model_backtest(
    symbol: str,
    timeframe: str,
    days: int,
    model_path: str,
    seq_len: int = 128,
    initial_equity: float = 10000.0,
) -> Tuple[BacktestResult, BacktestResult]:
    """
    返回 (model_result, teacher_result)
    """
    print(f"[V30 Eval] Loading raw OHLC: symbol={symbol}, timeframe={timeframe}, days={days}")
    df_raw = load_ohlc(symbol, timeframe, days)

    print("[V30 Eval] Running teacher_trend_follow...")
    teacher_cfg = TeacherConfig()
    df_teacher = teacher_trend_follow(df_raw, cfg=teacher_cfg)

    print("[V30 Eval] Building features...")
    df_feat, feat_cols = build_features(df_teacher)

    # 准备价格和时间
    prices = df_feat["close"].values.astype(np.float64)
    timestamps = df_feat.index.values

    # 老师动作（对照组）
    teacher_actions_full = df_feat["action_teacher"].astype(int).values

    # V30 模型策略动作（实验组）
    policy = V30TransformerPolicy(model_path=model_path)
    if policy.seq_len != seq_len:
        print(f"[V30 Eval] Warning: model seq_len={policy.seq_len}, override to {seq_len}")
        policy.seq_len = seq_len

    feats = df_feat[feat_cols].values.astype(np.float32)
    n = len(df_feat)
    if n < seq_len + 10:
        raise ValueError(f"Too few rows for backtest: n={n}, seq_len={seq_len}")

    model_actions = np.zeros(n, dtype=np.int64)

    print("[V30 Eval] Generating model actions over full series...")
    for i in range(seq_len - 1, n):
        start = i + 1 - seq_len
        seq_np = feats[start : i + 1]
        a = policy.predict_action(seq_np)
        model_actions[i] = a
        # 前 seq_len-1 部分留为 0 (HOLD)，影响很小

    print("[V30 Eval] Running backtest for model actions...")
    model_result = run_backtest_with_actions(
        prices=prices,
        timestamps=timestamps,
        actions=model_actions,
        initial_equity=initial_equity,
    )

    print("[V30 Eval] Running backtest for teacher actions...")
    teacher_result = run_backtest_with_actions(
        prices=prices,
        timestamps=timestamps,
        actions=teacher_actions_full,
        initial_equity=initial_equity,
    )

    return model_result, teacher_result


def print_bt_summary(name: str, bt: BacktestResult):
    print(f"===== {name} 回测结果 =====")
    print(f"最终权益: {bt.final_equity:.2f}")
    print(f"总收益率: {(bt.final_equity / bt.equity_curve[0] - 1.0) * 100:.2f}%")
    print(f"最大回撤: {bt.max_drawdown * 100:.2f}%")
    print(f"交易次数: {bt.num_trades}")
    print(f"胜率: {bt.win_rate * 100:.2f}%")
    print("===========================")


def parse_args():
    p = argparse.ArgumentParser(description="V30 Transformer 模型 + 老师策略 回测对比")
    p.add_argument("--symbol", type=str, default="BTCUSDT", help="交易对，例如 BTCUSDT")
    p.add_argument("--timeframe", type=str, default="1h", help="周期，例如 1h")
    p.add_argument("--days", type=int, default=2920, help="回看多少天历史数据")
    p.add_argument(
        "--model-path",
        type=str,
        default="",
        help="V30 Transformer 模型路径，默认使用 models_v30/v30_transformer_<symbol>_<timeframe>_best.pt",
    )
    p.add_argument("--seq-len", type=int, default=128, help="序列长度，需与训练时一致")
    p.add_argument("--initial-equity", type=float, default=10000.0, help="初始资金")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    symbol = args.symbol.upper()
    timeframe = args.timeframe
    days = args.days

    default_model_path = f"models_v30/v30_transformer_{symbol}_{timeframe}_best.pt"
    model_path = args.model_path or default_model_path

    print(f"[V30 Eval] Using model: {model_path}")
    model_bt, teacher_bt = run_v30_model_backtest(
        symbol=symbol,
        timeframe=timeframe,
        days=days,
        model_path=model_path,
        seq_len=args.seq_len,
        initial_equity=args.initial_equity,
    )

    print_bt_summary("V30 Transformer 模型", model_bt)
    print_bt_summary("Teacher 老师策略", teacher_bt)
