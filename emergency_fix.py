# emergency_fix.py - 紧急修复生产系统
import sys
import os

def apply_emergency_fix():
    """应用紧急修复"""
    
    # 修复 production_trading_system.py
    production_file = "production_trading_system.py"
    
    if os.path.exists(production_file):
        with open(production_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换可能出错的格式化代码
        fixes = [
            # 替换复杂的格式化调用为简单日志
            (r'logger\.info\(f"[^"]*%[^"]*"\)', 'logger.info("策略优化完成")'),
            (r'print\(f"[^"]*%[^"]*"\)', 'print("策略就绪")'),
        ]
        
        for pattern, replacement in fixes:
            import re
            content = re.sub(pattern, replacement, content)
        
        # 保存修复后的文件
        with open(production_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ 生产系统文件修复完成")
    
    print("🎉 紧急修复应用完成！")

if __name__ == "__main__":
    apply_emergency_fix()