# fix_multi_strategy.py
#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def fix_multi_strategy_manager():
    """修复多策略管理器的TradingSignal参数问题"""
    
    file_path = "multi_strategy_manager_enhanced.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找并替换有问题的代码段
        old_pattern = '''return TradingSignal(
            symbol=symbol,
            signal_type=final_type,
            price=avg_price,
            strength=min(final_strength, 1.0),
            timestamp=signals[0].timestamp,
            reason=reason,
            metadata=aggregation_metadata
        )'''
        
        new_code = '''return TradingSignal(
            symbol=symbol,
            signal_type=final_type,
            price=avg_price,
            strength=min(final_strength, 1.0),
            timestamp=signals[0].timestamp,
            reason=reason
        )'''
        
        if old_pattern in content:
            content = content.replace(old_pattern, new_code)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ 已成功修复 TradingSignal 参数问题")
            return True
        else:
            # 检查是否还有其他格式的相同问题
            import re
            pattern = r"return TradingSignal\([^)]*metadata\s*=[^)]+\)"
            matches = re.findall(pattern, content)
            if matches:
                for match in matches:
                    # 移除metadata参数
                    fixed_match = re.sub(r",\s*metadata\s*=[^,)]+", "", match)
                    content = content.replace(match, fixed_match)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print("✅ 已使用正则表达式修复 TradingSignal 参数问题")
                return True
            else:
                print("⚠️ 未找到需要修复的代码，可能已修复或代码格式不同")
                return False
                
    except Exception as e:
        print(f"❌ 修复过程中出错: {e}")
        return False

def fix_strategy_analyzer_imports():
    """修复策略分析器的导入问题"""
    
    file_path = "strategy_analyzer_simple.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 修复导入部分
        old_imports = '''from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig, DataManager
from src.backtesting.backtest_strategies import RobustSMAStrategy
from src.strategies.macd_strategy import MACDStrategy'''
        
        new_imports = '''from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig, DataManager
from src.backtesting.backtest_strategies import RobustSMAStrategy
from src.strategies.macd_strategy_smart import MACDStrategySmart
from src.strategies.strategy_orchestrator import BaseStrategy'''
        
        if old_imports in content:
            content = content.replace(old_imports, new_imports)
        
        # 注释掉有问题的导入
        if "from macd_strategy_debug import MACDStrategyDebug" in content:
            content = content.replace(
                "from macd_strategy_debug import MACDStrategyDebug", 
                "# from macd_strategy_debug import MACDStrategyDebug  # 已注释，使用智能版本"
            )
        
        # 更新策略配置
        old_strategies = '''    strategies = [
        ("SMA策略", RobustSMAStrategy, {"fast_period": 10, "slow_period": 30}),
        ("MACD标准", MACDStrategy, {"fast_period": 12, "slow_period": 26, "signal_period": 9}),
        ("MACD快速", MACDStrategy, {"fast_period": 6, "slow_period": 19, "signal_period": 5}),
    ]'''
        
        new_strategies = '''    strategies = [
        ("SMA策略", RobustSMAStrategy, {"fast_period": 10, "slow_period": 30}),
        ("MACD智能", MACDStrategySmart, {"fast_period": 12, "slow_period": 26, "signal_period": 9}),
    ]'''
        
        if old_strategies in content:
            content = content.replace(old_strategies, new_strategies)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ 已修复策略分析器导入问题")
        return True
        
    except Exception as e:
        print(f"❌ 修复策略分析器时出错: {e}")
        return False

def main():
    """执行所有修复"""
    print("🔧 开始修复AI量化交易系统...")
    print("=" * 50)
    
    success1 = fix_multi_strategy_manager()
    success2 = fix_strategy_analyzer_imports()
    
    print("=" * 50)
    if success1 and success2:
        print("🎉 所有修复完成！请重新运行测试。")
    else:
        print("⚠️ 部分修复可能未完成，请检查上述输出。")

if __name__ == "__main__":
    main()