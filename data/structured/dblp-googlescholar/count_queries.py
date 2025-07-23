import pandas as pd

# 读取 parquet 文件并打印行数
df = pd.read_parquet('sparkly_results_k50.parquet')
print(f"查询数量: {len(df)}")