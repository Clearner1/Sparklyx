import pandas as pd

# 检查sparkly_results_k50.parquet
print("=== Sparkly Results ===")
results = pd.read_parquet('sparkly_results_k50.parquet')
print("Shape:", results.shape)
print("Columns:", results.columns.tolist())
print("Sample rows:")
print(results.head(5))

# 检查结果中的_id是否在table_b范围内
print("\n=== Checking search results consistency ===")
id_in_range = results['_id'].isin(range(0, 64263))  # table_b的_id范围
print("_id values in table_b range:", id_in_range.all())

# 检查结果中的候选ID是否在table_a范围内
all_candidate_ids = set()
for idx, row in results.head(10).iterrows():  # 只检查前10行以节省时间
    all_candidate_ids.update(row['ids'])

candidate_in_range = [cid in range(0, 2616) for cid in list(all_candidate_ids)[:20]]  # 检查前20个
print("Candidate ids in table_a range (sample):", all(candidate_in_range))

# 查看一些具体的例子
print("\n=== Sample query details ===")
for idx, row in results.head(3).iterrows():
    query_id = row['_id']
    candidate_ids = row['ids'][:5]  # 只看前5个候选
    scores = row['scores'][:5]
    print(f"Query ID: {query_id}, Candidates: {candidate_ids}, Scores: {[round(s, 3) for s in scores]}")