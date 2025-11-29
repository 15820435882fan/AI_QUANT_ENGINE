# inspect_ai_optimizer.py
import inspect
from ai_strategy_optimizer_enhanced import *

# 查看模块中的所有类
print("🔍 AI优化器模块中的类:")
for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj):
        print(f"  ✅ {name}")

# 查看文件内容
with open('ai_strategy_optimizer_enhanced.py', 'r', encoding='utf-8') as f:
    content = f.read()
    # 查找类定义
    import re
    class_matches = re.findall(r'class\s+(\w+)', content)
    print(f"📋 文件中的类: {class_matches}")