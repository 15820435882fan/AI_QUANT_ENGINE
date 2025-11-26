# install_dependencies.py
import subprocess
import sys

def install_packages():
    """安装所需依赖包"""
    packages = [
        'ccxt',
        'pandas', 
        'numpy',
        'ta',  # 技术分析库
        'logging'
    ]
    
    for package in packages:
        try:
            print(f"📦 安装 {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} 安装成功")
        except subprocess.CalledProcessError:
            print(f"❌ {package} 安装失败")
    
    print("\n🎉 所有依赖安装完成！")

if __name__ == "__main__":
    install_packages()