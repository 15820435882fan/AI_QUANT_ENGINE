# setup_paths.py
#!/usr/bin/env python3
import sys
import os

def setup_project_paths():
    """设置项目路径，确保所有模块都能正确导入"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 添加项目根目录到Python路径
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # 添加src目录到Python路径
    src_dir = os.path.join(current_dir, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # 添加strategies目录到Python路径
    strategies_dir = os.path.join(current_dir, 'src', 'strategies')
    if strategies_dir not in sys.path:
        sys.path.insert(0, strategies_dir)
    
    print(f"✅ 路径设置完成:")
    print(f"   项目根目录: {current_dir}")
    print(f"   Python路径: {sys.path[:3]}")

# 立即执行路径设置
setup_project_paths()