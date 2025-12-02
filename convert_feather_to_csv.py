# convert_feather_to_csv.py
# 一键转换 data/feather 下的 feather 文件为 CSV
# 使用方法：
#   python convert_feather_to_csv.py
#
# 转换输出会保存在同目录，例如：
#   data/feather/BTC_USDT-15m.csv

import pandas as pd
import os
import sys

FEATHER_DIR = "data/feather"

def list_feather_files():
    return [f for f in os.listdir(FEATHER_DIR) if f.endswith(".feather")]

def convert_one(fname):
    fpath = os.path.join(FEATHER_DIR, fname)
    out_csv = os.path.join(FEATHER_DIR, fname.replace(".feather", ".csv"))

    print(f"正在读取 feather 文件：{fpath}")

    try:
        df = pd.read_feather(fpath)
    except ImportError:
        print("❌ 当前环境没有安装 pyarrow，无法读取 feather 文件")
        print("👉 请在你的系统中执行：pip install pyarrow")
        sys.exit(1)

    print(f"读取成功，开始写入 CSV：{out_csv}")
    df.to_csv(out_csv, index=False)
    print("🎉 转换成功！")

def main():
    print("扫描 data/feather 目录中的 feather 文件...\n")
    files = list_feather_files()

    if not files:
        print("❌ 没有找到任何 feather 文件，请检查 data/feather 目录。")
        return

    print("找到以下 feather 文件：")
    for idx, f in enumerate(files):
        print(f"{idx+1}. {f}")

    print("\n请输入要转换的文件编号（数字）：")
    try:
        sel = int(input("> "))
        fname = files[sel - 1]
    except:
        print("输入错误")
        return

    convert_one(fname)

if __name__ == "__main__":
    main()
