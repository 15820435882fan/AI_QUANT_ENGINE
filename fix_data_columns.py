# 创建紧急修复文件：fix_data_columns.py
import pandas as pd
import numpy as np

def ensure_required_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    确保数据包含策略所需的所有列
    """
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # 检查并重命名列（如果存在大小写问题）
    column_mapping = {}
    for req_col in required_columns:
        if req_col not in data.columns:
            # 检查大写版本
            upper_col = req_col.upper()
            if upper_col in data.columns:
                column_mapping[upper_col] = req_col
            else:
                # 如果列不存在，创建默认值
                if req_col in ['open', 'high', 'low', 'close']:
                    data[req_col] = data.iloc[:, 0] if len(data.columns) > 0 else 1.0
                elif req_col == 'volume':
                    data[req_col] = 1.0
    
    if column_mapping:
        data = data.rename(columns=column_mapping)
    
    return data

# 立即测试修复
if __name__ == "__main__":
    # 模拟问题数据
    test_data = pd.DataFrame({
        'OPEN': [1, 2, 3],
        'CLOSE': [1.1, 2.1, 3.1]
    })
    
    fixed_data = ensure_required_columns(test_data)
    print("修复前列名:", test_data.columns.tolist())
    print("修复后列名:", fixed_data.columns.tolist())
    print("✅ 数据列修复完成")