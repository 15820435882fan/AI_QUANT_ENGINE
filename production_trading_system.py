# production_trading_system.py - 生产环境交易主系统（集成组合打分 + 参数优化）

import pandas as pd
import numpy as np
import logging
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

from multi_strategy_manager_enhanced import MultiStrategyManagerEnhanced
from ai_strategy_optimizer_enhanced import EnhancedAIStrategyOptimizer
from multi_strategy_optimizer import MultiStrategyOptimizer


class ProductionTradingSystem:
    """生产环境交易系统（多策略组合 + AI 参数优化）"""

    def __init__(self, use_combo_optimizer: bool = True):
        # 多策略管理器（负责把所有策略统一成 combined_signal）
        self.strategy_manager = MultiStrategyManagerEnhanced()

        # 单策略参数优化器（遗传算法那一套）
        self.optimizer = EnhancedAIStrategyOptimizer()

        # 策略组合优化器（用来挑选“哪几种策略一起上场”）
        self.use_combo_optimizer = use_combo_optimizer
        self.combo_optimizer: Optional[MultiStrategyOptimizer] = (
            MultiStrategyOptimizer() if use_combo_optimizer else None
        )

        # 记录当前使用的优化后策略
        self.optimized_strategies: Dict[str, Dict[str, Any]] = {}

        # 最近一次优化时间
        self.last_optimization_time: Optional[datetime] = None

        self.setup_logging()

    # ------------------------------------------------------------------ #
    # 日志配置
    # ------------------------------------------------------------------ #
    def setup_logging(self):
        """设置生产环境日志（文件 + 控制台，统一 UTF-8 编码）"""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # 文件日志
        file_handler = logging.FileHandler("trading_system.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        # 控制台日志
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 清理旧 handler，避免重复输出
        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        self.logger = logging.getLogger(__name__)
        self.logger.info("✅ ProductionTradingSystem 日志系统初始化完成")

    # ------------------------------------------------------------------ #
    # 工具：参数名兼容处理
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_params_for_strategy(
        strategy_type: str, params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        把 AI 优化器返回的参数名转换为各策略真正需要的名字。

        不改策略源码，只在这里做一层兼容映射，避免出现“缺少参数”告警。
        """
        if not params:
            return {}

        normalized = dict(params)

        # MACDStrategySmart: 期望 fast_period / slow_period / signal_period
        if strategy_type == "MACDStrategySmart":
            if "macd_fast" in params:
                normalized["fast_period"] = params["macd_fast"]
            if "macd_slow" in params:
                normalized["slow_period"] = params["macd_slow"]
            if "macd_signal" in params:
                normalized["signal_period"] = params["macd_signal"]

        # BollingerBandsStrategy: 期望 period / std_dev
        if strategy_type == "BollingerBandsStrategy":
            if "bb_period" in params:
                normalized["period"] = params["bb_period"]
            if "bb_std" in params:
                normalized["std_dev"] = params["bb_std"]

        return normalized

    # ------------------------------------------------------------------ #
    # 策略初始化 & 优化
    # ------------------------------------------------------------------ #
    def _get_default_combinations(self) -> List[List[str]]:
        """给一个默认的组合列表，供组合优化器选择"""
        return [
            ["SimpleMovingAverageStrategy", "MACDStrategySmart"],
            ["SimpleMovingAverageStrategy", "BollingerBandsStrategy"],
            ["MACDStrategySmart", "BollingerBandsStrategy"],
            [
                "SimpleMovingAverageStrategy",
                "MACDStrategySmart",
                "BollingerBandsStrategy",
            ],
        ]

    def initialize_optimized_strategies(
        self,
        historical_data: pd.DataFrame,
        strategy_combinations: Optional[List[List[str]]] = None,
    ):
        """
        初始化并优化策略：

        1）先用 MultiStrategyOptimizer 选出“最佳策略组合”；
        2）对组合里的每个策略分别做参数优化；
        3）把优化后的策略挂到 MultiStrategyManagerEnhanced 里。
        """
        self.logger.info("🧠 初始化 & 优化策略组合...")

        if historical_data is None or historical_data.empty:
            self.logger.error("历史数据为空，无法初始化策略")
            raise ValueError("historical_data 不能为空")

        if strategy_combinations is None:
            strategy_combinations = self._get_default_combinations()

        # 1. 组合打分：选出哪几种策略一起上
        if self.use_combo_optimizer and self.combo_optimizer is not None:
            self.logger.info("📊 使用 MultiStrategyOptimizer 评估策略组合...")
            (
                best_combination,
                best_score,
            ) = self.combo_optimizer.optimize_strategy_combination(
                historical_data,
                strategy_combinations=strategy_combinations,
                evaluator=None,  # 使用 MultiStrategyOptimizer 内置打分逻辑
            )
            if not best_combination:
                self.logger.warning(
                    "组合优化器没有给出可用组合，回退到默认组合 SimpleMovingAverage + MACD"
                )
                best_combination = ["SimpleMovingAverageStrategy", "MACDStrategySmart"]
                best_score = None
        else:
            best_combination = ["SimpleMovingAverageStrategy", "MACDStrategySmart"]
            best_score = None

        self.logger.info(
            f"🎯 最终选择的策略组合: {best_combination}（score={best_score}）"
        )

        # 先清空原有策略，避免重复
        self.strategy_manager.strategies.clear()
        self.optimized_strategies.clear()

        # 2. 对组合中的每个策略做参数优化 + 注册进 manager
        for strategy_type in best_combination:
            try:
                self.logger.info(f"🛠 开始优化策略: {strategy_type} ...")
                (
                    best_params,
                    best_score_single,
                ) = self.optimizer.optimize_strategy_parameters(
                    strategy_type,
                    historical_data,
                    generations=8,  # 适当缩小，方便你快速验证
                    population_size=10,
                )

                # 做一层参数名兼容处理，避免 MACD/BOLL 参数不匹配
                normalized_params = self._normalize_params_for_strategy(
                    strategy_type, best_params
                )

                config = {
                    "name": f"优化_{strategy_type}",
                    "parameters": normalized_params,
                }
                strategy_instance = self.strategy_manager.add_strategy(
                    strategy_type, config
                )

                if strategy_instance is None:
                    self.logger.error(f"❌ {strategy_type} 添加失败，跳过")
                    continue

                self.optimized_strategies[strategy_type] = {
                    "strategy": strategy_instance,
                    "parameters": normalized_params,
                    "score": best_score_single,
                }
                self.logger.info(
                    f"✅ {strategy_type} 优化完成: score={best_score_single}, params={normalized_params}"
                )

            except Exception as e:
                self.logger.error(f"❌ {strategy_type} 优化失败: {e}")

        self.last_optimization_time = datetime.now()
        self.logger.info(
            f"🎉 策略初始化完毕，当前有效策略数: {len(self.strategy_manager.strategies)}"
        )

    # ------------------------------------------------------------------ #
    # 实时行情处理 & 交易决策
    # ------------------------------------------------------------------ #
    def process_market_data(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """处理市场数据并生成交易决策"""
        self.logger.info(f"📨 收到市场数据，形状: {market_data.shape}")

        try:
            combined_signals = self.strategy_manager.calculate_combined_signals(
                market_data
            )

            if combined_signals.empty:
                self.logger.warning("⚠️ 当前没有任何有效信号")
                return {"error": "无有效信号", "action": "HOLD"}

            latest_signal = float(combined_signals["combined_signal"].iloc[-1])
            decision = self._make_trading_decision(latest_signal, combined_signals)

            self.logger.info(f"📤 交易决策: {decision}")
            return decision

        except Exception as e:
            self.logger.error(f"信号处理失败: {e}")
            return {"error": str(e), "action": "HOLD"}

    def _make_trading_decision(
        self, signal: float, signals_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """基于组合信号生成最终交易决策"""

        # 信号强度 + 近期趋势
        signal_strength = abs(signal)
        recent_trend = float(signals_df["combined_signal"].tail(10).mean())

        # 阈值可以后面做成配置或再交给 AI 优化
        strong_threshold = 0.5
        weak_threshold = 0.3

        if signal > strong_threshold:
            action = "BUY"
            confidence = min(signal_strength * 2.0, 1.0)
        elif signal < -strong_threshold:
            action = "SELL"
            confidence = min(signal_strength * 2.0, 1.0)
        elif signal > weak_threshold:
            action = "BUY"
            confidence = 0.4
        elif signal < -weak_threshold:
            action = "SELL"
            confidence = 0.4
        else:
            action = "HOLD"
            confidence = 0.1

        return {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "confidence": round(confidence, 3),
            "signal_strength": round(signal_strength, 3),
            "signal_trend": round(recent_trend, 3),
            "strategies_used": len(self.strategy_manager.strategies),
            "optimized_strategies": list(self.optimized_strategies.keys()),
        }

    # ------------------------------------------------------------------ #
    # 系统状态
    # ------------------------------------------------------------------ #
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态（给监控 / dashboard 用）"""
        strategies_info = self.strategy_manager.get_strategies_info()

        return {
            "status": "RUNNING",
            "active_strategies": len(self.strategy_manager.strategies),
            "optimized_strategies": list(self.optimized_strategies.keys()),
            "last_optimization": self.last_optimization_time.isoformat()
            if self.last_optimization_time
            else None,
            "strategies_detail": strategies_info,
        }


# ---------------------------------------------------------------------- #
# 自测入口（不影响生产引用）
# ---------------------------------------------------------------------- #
def test_production_system() -> ProductionTradingSystem:
    """本地快速测试生产交易系统"""
    print("🧪 测试 ProductionTradingSystem ...")

    from test_strategies_with_real_data import generate_realistic_test_data

    trading_system = ProductionTradingSystem()

    # 用较少的历史数据快速跑一轮优化
    historical_data = generate_realistic_test_data(200)
    trading_system.initialize_optimized_strategies(historical_data)

    # 模拟实时数据
    realtime_data = generate_realistic_test_data(50)
    decision = trading_system.process_market_data(realtime_data)
    print(f"交易决策: {decision}")

    status = trading_system.get_system_status()
    print(f"系统状态: {status}")

    return trading_system


if __name__ == "__main__":
    test_production_system()
