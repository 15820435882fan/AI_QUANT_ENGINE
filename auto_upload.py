import subprocess
import datetime

def auto_upload():
    try:
        print("🚀 开始自动上传到GitHub...")
        
        # 添加所有文件
        subprocess.run(["git", "add", "."], check=True)
        print("✅ 文件已添加")
        
        # 提交
        commit_msg = f"自动提交: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        print("✅ 提交完成")
        
        # 推送
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("🎉 代码上传成功!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 上传失败: {e}")

if __name__ == "__main__":
    auto_upload()