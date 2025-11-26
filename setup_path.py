# setup_path.py
import sys
import os

def setup_project_path():
    """设置项目路径 - 在所有文件开头调用"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"✅ 路径设置完成: {project_root}")

# 立即执行
setup_project_path()