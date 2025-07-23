import pandas as pd

# 检查gold.parquet和gold.json是否一致
print("=== Comparing gold.parquet and gold.json ===")
gold_parquet = pd.read_parquet('gold.parquet')
with open('gold.json', 'r') as f:
    import json
    gold_json = pd.DataFrame(json.load(f))

print("gold.parquet shape:", gold_parquet.shape)
print("gold.json shape:", gold_json.shape)

print("gold.parquet columns:", gold_parquet.columns.tolist())
print("gold.json columns:", gold_json.columns.tolist())

print("gold.parquet head:")
print(gold_parquet.head())

print("gold.json head:")
print(gold_json.head())

# 检查是否完全相同
print("\n=== Checking if they are identical ===")
same_shape = gold_parquet.shape == gold_json.shape
same_columns = gold_parquet.columns.tolist() == gold_json.columns.tolist()

if same_shape and same_columns:
    # 比较内容
    identical = gold_parquet.equals(gold_json)
    print("Identical content:", identical)
else:
    print("Different shape or columns")

# 检查是否有重复的映射
print("\n=== Checking for duplicate mappings ===")
duplicates_parquet = gold_parquet.duplicated().sum()
duplicates_json = gold_json.duplicated().sum()
print("Duplicate rows in gold.parquet:", duplicates_parquet)
print("Duplicate rows in gold.json:", duplicates_json)

# 检查一个具体的gold条目是否在table_a和table_b中存在
print("\n=== Checking specific gold mappings ===")
sample_gold = gold_parquet.head(5)
table_a = pd.read_parquet('table_a.parquet')
table_b = pd.read_parquet('table_b.parquet')

for idx, row in sample_gold.iterrows():
    id1, id2 = row['id1'], row['id2']
    in_table_a = id1 in table_a['_id'].values
    in_table_b = id2 in table_b['_id'].values
    print(f"Gold mapping {id1}->{id2}: id1 in table_a={in_table_a}, id2 in table_b={in_table_b}")