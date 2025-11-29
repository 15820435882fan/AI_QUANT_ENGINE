# find_correct_methods.py
"""
直接检查各个模块的正确方法名
"""
import inspect
from real_market_data import RealMarketData
from enhanced_sniper_detector import EnhancedSniperDetector
from ai_strategy_optimizer_enhanced import EnhancedAIStrategyOptimizer
from production_trading_system import ProductionTradingSystem

print("🔍 检查各个模块的方法...")

# 1. 检查RealMarketData
print("\n📊 RealMarketData 方法:")
market_data = RealMarketData()
for method in dir(market_data):
    if not method.startswith('_'):
        print(f"  ✅ {method}")

# 2. 检查EnhancedSniperDetector  
print("\n🎯 EnhancedSniperDetector 方法:")
detector = EnhancedSniperDetector()
for method in dir(detector):
    if not method.startswith('_'):
        print(f"  ✅ {method}")

# 3. 检查EnhancedAIStrategyOptimizer
print("\n🤖 EnhancedAIStrategyOptimizer 方法:")
optimizer = EnhancedAIStrategyOptimizer()
for method in dir(optimizer):
    if not method.startswith('_'):
        print(f"  ✅ {method}")

# 4. 检查ProductionTradingSystem
print("\n⚡ ProductionTradingSystem 方法:")
production = ProductionTradingSystem()
for method in dir(production):
    if not method.startswith('_'):
        print(f"  ✅ {method}")

print("\n🎯 基于健康检查代码推断方法...")
# 查看健康检查中如何使用这些模块
with open('system_health_check_final.py', 'r', encoding='utf-8') as f:
    content = f.read()
    # 查找方法调用模式
    import re
    method_calls = re.findall(r'\.(\w+)\s*\(', content)
    print("健康检查中的方法调用:", set(method_calls))