import json
import pandas as pd

print("=== 检查数据匹配情况 ===")

# 检查完整数据
print("\n📊 完整数据统计:")
df_a = pd.read_parquet('table_a.parquet')
df_b = pd.read_parquet('table_b.parquet')
print(f"Table A: {len(df_a)} 条记录, ID范围: {df_a['_id'].min()} - {df_a['_id'].max()}")
print(f"Table B: {len(df_b)} 条记录, ID范围: {df_b['_id'].min()} - {df_b['_id'].max()}")

# 检查搜索结果
print("\n🔍 搜索结果分析:")
with open('sparkly_results_k50.md', 'r', encoding='utf-8') as f:
    result = json.loads(f.read().strip())

query_id = result['_id']
candidate_ids = result['ids']
print(f"查询ID: {query_id}")
print(f"候选数量: {len(candidate_ids)}")
print(f"候选ID范围: {min(candidate_ids)} - {max(candidate_ids)}")
print(f"前10个候选: {candidate_ids[:10]}")

# 检查候选ID是否都在Table A中
missing_ids = []
for cand_id in candidate_ids:
    if cand_id not in df_a['_id'].values:
        missing_ids.append(cand_id)

if missing_ids:
    print(f"⚠️ 警告: {len(missing_ids)} 个候选ID在Table A中不存在")
    print(f"缺失的ID样例: {missing_ids[:5]}")
else:
    print("✅ 所有候选ID都在Table A中存在")

# 检查查询ID是否在Table B中
if query_id in df_b['_id'].values:
    print(f"✅ 查询ID {query_id} 在Table B中存在")
else:
    print(f"⚠️ 查询ID {query_id} 在Table B中不存在")

# 检查真实标签
print("\n🎯 真实标签检查:")
with open('gold_part.md', 'r', encoding='utf-8') as f:
    gold_lines = [line.strip() for line in f if line.strip()]

gold_mapping = {}
for line in gold_lines:
    data = json.loads(line)
    gold_mapping[data['id2']] = data['id1']

if query_id in gold_mapping:
    true_match = gold_mapping[query_id]
    print(f"✅ 查询ID {query_id} 的真实匹配: Table A ID {true_match}")
    
    if true_match in candidate_ids:
        rank = candidate_ids.index(true_match) + 1
        print(f"✅ 真实匹配在候选列表第 {rank} 位")
    else:
        print("❌ 真实匹配不在候选列表中")
else:
    print(f"❌ 查询ID {query_id} 没有真实标签") 