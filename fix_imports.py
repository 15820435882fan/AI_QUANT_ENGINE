# fix_imports.py
import sys
import os

def setup_environment():
    """设置项目环境路径"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    print(f"✅ 项目根目录已添加到路径: {project_root}")

# 立即执行
setup_environment()