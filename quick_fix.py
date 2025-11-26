# quick_fix.py
import sys
import os

def check_imports():
    """检查所有必要的导入"""
    try:
        from typing import Dict, List, Any
        import pandas as pd
        import numpy as np
        import ta
        import logging
        from datetime import datetime, timedelta
        
        print("✅ 所有基础导入成功")
        
        # 检查自定义模块
        try:
            from sniper_signal_detector import SniperSignalDetector
            print("✅ sniper_signal_detector 导入成功")
        except Exception as e:
            print(f"❌ sniper_signal_detector 导入失败: {e}")
            
        try:
            from enhanced_sniper_detector import EnhancedSniperDetector
            print("✅ enhanced_sniper_detector 导入成功")
        except Exception as e:
            print(f"❌ enhanced_sniper_detector 导入失败: {e}")
            
        try:
            from sniper_position_manager import SniperPositionManager
            print("✅ sniper_position_manager 导入成功")
        except Exception as e:
            print(f"❌ sniper_position_manager 导入失败: {e}")
            
        try:
            from advanced_position_manager import AdvancedPositionManager
            print("✅ advanced_position_manager 导入成功")
        except Exception as e:
            print(f"❌ advanced_position_manager 导入失败: {e}")
            
        try:
            from trade_analyzer import TradeAnalyzer
            print("✅ trade_analyzer 导入成功")
        except Exception as e:
            print(f"❌ trade_analyzer 导入失败: {e}")
            
    except Exception as e:
        print(f"❌ 基础导入失败: {e}")

if __name__ == "__main__":
    print("🔍 检查系统导入状态...")
    check_imports()