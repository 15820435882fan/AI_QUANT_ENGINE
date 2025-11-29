# multi_strategy_optimizer.py
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Callable, Tuple, Optional

from multi_strategy_manager_enhanced import MultiStrategyManagerEnhanced


ScoreEvaluator = Callable[
    [MultiStrategyManagerEnhanced, pd.DataFrame, List[str]],
    Tuple[float, Dict[str, Any]],
]


class MultiStrategyOptimizer:
    """多策略组合优化器（支持 AI/回测打分）"""

    def __init__(self):
        self.manager = MultiStrategyManagerEnhanced()
        # key: 组合(str(tuple(...))) -> {'score': float, 'metrics': dict}
        self.optimization_results: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # 对外主接口
    # ------------------------------------------------------------------ #
    def optimize_strategy_combination(
        self,
        data: pd.DataFrame,
        strategy_combinations: List[List[str]],
        evaluator: Optional[ScoreEvaluator] = None,
    ) -> Tuple[List[str], float]:
        """
        优化策略组合。

        参数：
        - data: 历史数据（至少包含 'close'，其他列由 manager 做预处理）
        - strategy_combinations: [['SimpleMovingAverageStrategy', 'MACDStrategySmart'], ...]
        - evaluator: 可选打分函数，如果不给，就用内置评分逻辑。

        返回：
        - best_combination: 得分最高的策略类型列表
        - best_score: 对应得分
        """
        print("🧬 开始多策略组合优化...")

        best_combination: List[str] | None = None
        best_score: float = -np.inf

        for i, combination in enumerate(strategy_combinations):
            print(f"\n🔍 测试组合 {i + 1}/{len(strategy_combinations)}: {combination}")

            # 清空当前策略
            self.manager.strategies.clear()

            # 添加策略组合
            for strategy_type in combination:
                config = self._get_default_config(strategy_type)
                self.manager.add_strategy(strategy_type, config)

            # 评估组合
            if evaluator is not None:
                score, metrics = evaluator(self.manager, data, combination)
            else:
                score, metrics = self._evaluate_combination_default(data)

            combo_key = str(tuple(combination))
            self.optimization_results[combo_key] = {
                "score": score,
                "metrics": metrics,
            }

            if score > best_score:
                best_score = score
                best_combination = combination

            print(f"📊 组合 {combination} 得分: {score:.4f}, metrics={metrics}")

        print(f"\n🎯 最佳策略组合: {best_combination}")
        print(f"📊 最佳得分: {best_score:.4f}")

        return best_combination or [], best_score

    # ------------------------------------------------------------------ #
    # 默认参数 & 默认评分逻辑
    # ------------------------------------------------------------------ #
    def _get_default_config(self, strategy_type: str) -> Dict[str, Any]:
        """为每种策略给一份简单的默认参数"""
        default_configs: Dict[str, Dict[str, Any]] = {
            "SimpleMovingAverageStrategy": {
                "name": f"{strategy_type}_默认",
                "parameters": {"sma_fast": 10, "sma_slow": 30},
            },
            "MACDStrategySmart": {
                "name": f"{strategy_type}_默认",
                "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
            },
            "BollingerBandsStrategy": {
                "name": f"{strategy_type}_默认",
                "parameters": {"period": 20, "std_dev": 2.0},
            },
        }
        return default_configs.get(strategy_type, {"name": strategy_type, "parameters": {}})

    def _evaluate_combination_default(
        self, data: pd.DataFrame
    ) -> Tuple[float, Dict[str, Any]]:
        """
        默认评分逻辑：

        1）manager.calculate_combined_signals -> combined_signal;
        2）用 combined_signal 在 close 上合成一个粗糙的收益曲线；
        3）综合：
           - 信号方差（希望有变化但不要太平）；
           - 信号切换次数；
           - 简单的总收益 & “Sharpe 近似”。

        返回：
        - score: 一个综合分
        - metrics: 中间指标（方便后续调参）
        """
        try:
            combined_signals = self.manager.calculate_combined_signals(data)
            if combined_signals.empty:
                return -np.inf, {"reason": "no_signals"}

            if "combined_signal" not in combined_signals.columns:
                return -np.inf, {"reason": "no_combined_signal"}

            signal_series = combined_signals["combined_signal"].astype(float)

            # 1. 信号质量（方差 + 变化次数）
            signal_variance = float(signal_series.var())
            signal_changes = int((signal_series.diff().fillna(0) != 0).sum())

            # 2. 简单收益曲线（直接用 combined_signal 作为仓位 [-1,1]）
            if "close" not in data.columns:
                equity_metrics = {"total_return": 0.0, "sharpe_like": 0.0}
            else:
                close = data["close"].astype(float)
                ret = close.pct_change().fillna(0.0)

                # 仓位 = 上一时刻的 combined_signal，避免未来函数
                position = signal_series.shift(1).fillna(0.0).clip(-1.0, 1.0)

                strat_ret = position * ret
                equity = (1 + strat_ret).cumprod()

                total_return = float(equity.iloc[-1] - 1.0)
                if strat_ret.std() > 1e-8:
                    sharpe_like = float(strat_ret.mean() / strat_ret.std() * np.sqrt(252))
                else:
                    sharpe_like = 0.0

                equity_metrics = {
                    "total_return": total_return,
                    "sharpe_like": sharpe_like,
                }

            # 综合评分：
            # - 控制一下信号变化数，太多/太少都不好
            stability_score = signal_variance
            change_penalty = abs(signal_changes - 20) * 0.02  # 希望变化数量在 20 左右

            score = (
                stability_score * 0.4
                + equity_metrics["total_return"] * 0.4
                + equity_metrics["sharpe_like"] * 0.3
                - change_penalty
            )

            metrics = {
                "signal_variance": signal_variance,
                "signal_changes": signal_changes,
                **equity_metrics,
            }
            return float(score), metrics

        except Exception as e:
            print(f"⚠️ 默认组合评估失败: {e}")
            return -np.inf, {"reason": f"exception: {e}"}


# ---------------------------------------------------------------------- #
# 自测入口
# ---------------------------------------------------------------------- #
def test_multi_strategy_optimizer() -> MultiStrategyOptimizer:
    """测试多策略优化器（仍然使用你原来的 generate_realistic_test_data）"""
    print("🚀 测试多策略组合优化器...")

    from test_strategies_with_real_data import generate_realistic_test_data

    test_data = generate_realistic_test_data(150)

    optimizer = MultiStrategyOptimizer()

    strategy_combinations = [
        ["SimpleMovingAverageStrategy", "MACDStrategySmart"],
        ["SimpleMovingAverageStrategy", "BollingerBandsStrategy"],
        ["MACDStrategySmart", "BollingerBandsStrategy"],
        [
            "SimpleMovingAverageStrategy",
            "MACDStrategySmart",
            "BollingerBandsStrategy",
        ],
    ]

    best_combination, best_score = optimizer.optimize_strategy_combination(
        test_data, strategy_combinations
    )

    print("\n📊 所有组合结果:")
    for combo, res in optimizer.optimization_results.items():
        print(f"  {combo}: score={res['score']:.4f}, metrics={res['metrics']}")

    return optimizer


if __name__ == "__main__":
    test_multi_strategy_optimizer()
