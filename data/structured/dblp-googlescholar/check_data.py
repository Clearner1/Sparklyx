import pandas as pd

# 检查gold.parquet
print("=== Gold file ===")
gold = pd.read_parquet('gold.parquet')
print("Shape:", gold.shape)
print("Columns:", gold.columns.tolist())
print("Head:")
print(gold.head(10))

print("\n=== Table A ===")
table_a = pd.read_parquet('table_a.parquet')
print("Shape:", table_a.shape)
print("Columns:", table_a.columns.tolist())
print("_id range:", table_a['_id'].min(), "to", table_a['_id'].max())

print("\n=== Table B ===")
table_b = pd.read_parquet('table_b.parquet')
print("Shape:", table_b.shape)
print("Columns:", table_b.columns.tolist())
print("_id range:", table_b['_id'].min(), "to", table_b['_id'].max())

print("\n=== Checking gold mapping consistency ===")
# 检查id1是否在table_a的_id范围内
id1_in_range = gold['id1'].isin(table_a['_id'])
print("id1 values in table_a range:", id1_in_range.all())

# 检查id2是否在table_b的_id范围内
id2_in_range = gold['id2'].isin(table_b['_id'])
print("id2 values in table_b range:", id2_in_range.all())

# 查看不一致的记录
if not id1_in_range.all():
    print("Invalid id1 values:")
    invalid_id1 = gold[~id1_in_range]
    print(invalid_id1)

if not id2_in_range.all():
    print("Invalid id2 values:")
    invalid_id2 = gold[~id2_in_range]
    print(invalid_id2)