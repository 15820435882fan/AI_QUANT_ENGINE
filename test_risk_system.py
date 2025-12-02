# test_risk_system.py
#!/usr/bin/env python3
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.risk.risk_manager import RiskManager, RiskConfig

async def test_risk_system():
    """测试风险管理系统"""
    print("🧪 测试风险管理系统...")
    
    risk_manager = RiskManager(RiskConfig())
    
    # 测试交易验证
    test_signal = {
        'action': 'buy',
        'price': 50000.0,
        'quantity': 0.1
    }
    
    risk_result = await risk_manager.validate_trade(
        test_signal, 
        current_equity=10000.0,
        positions={},
        today_trades=5
    )
    
    print(f"📊 风险检查结果: {risk_result['approved']}")
    print(f"💡 原因: {risk_result['reason']}")
    
    if risk_result['adjusted_quantity']:
        print(f"🎯 建议仓位: {risk_result['adjusted_quantity']:.4f}")
    
    # 测试盈亏更新
    risk_manager.update_pnl(-150)  # 模拟亏损
    print(f"💰 当日盈亏: {risk_manager.daily_pnl:.2f}")
    
    print("✅ 风险系统测试完成")

if __name__ == "__main__":
    asyncio.run(test_risk_system())