#!/usr/bin/env python3
"""
SimHash重排序演示脚本

使用abt_buy数据集演示SimHash重排序的完整流程：
1. 加载数据和配置
2. 对样例查询进行重排序
3. 比较重排序前后的效果
4. 输出详细的分析报告

运行方式: python demo_reranking.py
"""

import json
import pandas as pd
import numpy as np
from simhash_reranker import SimHashReranker, RerankerEvaluator, RerankerConfig


def load_table_data(table_path: str) -> pd.DataFrame:
    """加载表格数据"""
    if table_path.endswith('.parquet'):
        return pd.read_parquet(table_path)
    else:
        records = []
        with open(table_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))
        return pd.DataFrame(records)


def load_search_results(results_path: str) -> dict:
    """加载搜索结果"""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.loads(f.read().strip())


def print_record_details(record: dict, title: str):
    """打印记录详细信息"""
    print(f"\n=== {title} ===")
    print(f"ID: {record['_id']}")
    print(f"Name: {record['name']}")
    print(f"Description: {record['description'][:100]}...")
    print(f"Price: {record['price']}")


def main():
    print("🚀 SimHash重排序演示开始")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n📁 加载数据...")
    table_a = load_table_data('table_a.parquet')  # 使用完整数据
    table_b = load_table_data('table_b.parquet')  # 使用完整数据
    search_results = load_search_results('sparkly_results_k50.md')
    
    print(f"✅ Table A: {len(table_a)} 条记录")
    print(f"✅ Table B: {len(table_b)} 条记录")
    print(f"✅ 搜索结果已加载")
    
    # 2. 创建重排序器并加载配置
    print("\n⚙️ 初始化重排序器...")
    config = RerankerConfig(
        simhash_bits=64,
        alpha=0.7,      # BM25权重
        beta=0.3,       # SimHash权重
        use_3gram=True,
        normalize_scores=True
    )
    
    reranker = SimHashReranker(config)
    reranker.load_optimization_result('optimization_result.json')
    
    # 3. 创建评估器
    evaluator = RerankerEvaluator()
    gold_mapping = evaluator.load_ground_truth('gold_part.md')
    print(f"✅ 加载了 {len(gold_mapping)} 个真实标签")
    
    # 4. 分析演示案例
    query_id = search_results['_id']  # 查询ID=2
    candidate_ids = search_results['ids']
    original_scores = search_results['scores']
    
    print(f"\n🔍 分析查询案例 (ID={query_id})")
    
    # 获取查询记录
    query_record = table_b[table_b['_id'] == query_id].iloc[0].to_dict()
    print_record_details(query_record, "查询记录")
    
    # 获取候选记录
    candidate_records = []
    for cand_id in candidate_ids:
        candidate_record = table_a[table_a['_id'] == cand_id].iloc[0].to_dict()
        candidate_records.append(candidate_record)
    
    # 显示原始排序前5名
    print(f"\n📊 原始BM25排序 (前5名):")
    for i in range(5):
        cand_id = candidate_ids[i]
        score = original_scores[i]
        record = candidate_records[i]
        is_correct = (query_id in gold_mapping and gold_mapping[query_id] == cand_id)
        marker = "✅ 正确答案!" if is_correct else ""
        print(f"  {i+1}. ID={cand_id:<4} 分数={score:6.2f} {marker}")
        print(f"     名称: {record['name'][:50]}...")
    
    # 5. 执行重排序
    print(f"\n🔄 执行SimHash重排序...")
    ranking_indices, fused_scores, debug_info = reranker.rerank_candidates(
        query_record, candidate_records, original_scores
    )
    
    # 重新排列候选ID
    reranked_ids = [candidate_ids[i] for i in ranking_indices]
    
    # 显示重排序结果前5名
    print(f"\n🎯 SimHash重排序结果 (前5名):")
    for i in range(5):
        orig_idx = ranking_indices[i]
        cand_id = candidate_ids[orig_idx]
        fused_score = fused_scores[i]
        record = candidate_records[orig_idx]
        is_correct = (query_id in gold_mapping and gold_mapping[query_id] == cand_id)
        marker = "✅ 正确答案!" if is_correct else ""
        
        orig_rank = candidate_ids.index(cand_id) + 1
        rank_change = f"(原排名#{orig_rank})" if orig_rank != i+1 else ""
        
        print(f"  {i+1}. ID={cand_id:<4} 分数={fused_score:6.3f} {rank_change} {marker}")
        print(f"     名称: {record['name'][:50]}...")
    
    # 6. 评估效果
    print(f"\n📈 效果评估:")
    metrics = evaluator.calculate_metrics(
        query_id, candidate_ids, reranked_ids, gold_mapping
    )
    
    if metrics['has_ground_truth']:
        true_id = metrics['true_match_id']
        print(f"✅ 真实匹配: Table A ID = {true_id}")
        
        for k in [1, 5, 10]:
            orig_recall = metrics[f'original_recall@{k}']
            rerank_recall = metrics[f'reranked_recall@{k}']
            orig_rank = metrics[f'original_rank@{k}']
            rerank_rank = metrics[f'reranked_rank@{k}']
            improvement = metrics[f'rank_improvement@{k}']
            
            print(f"\n  📊 Recall@{k}:")
            print(f"     原始: {orig_recall:.3f} (排名: {orig_rank if orig_rank > 0 else 'N/A'})")
            print(f"     重排: {rerank_recall:.3f} (排名: {rerank_rank if rerank_rank > 0 else 'N/A'})")
            if improvement > 0:
                print(f"     🎉 排名提升: +{improvement} 位")
            elif improvement < 0:
                print(f"     ⚠️ 排名下降: {improvement} 位")
    else:
        print("❌ 该查询没有真实标签")
    
    # 7. 技术细节分析
    print(f"\n🔧 技术细节:")
    print(f"  处理时间: {debug_info['processing_time']:.4f} 秒")
    print(f"  SimHash位数: {debug_info['config']['simhash_bits']}")
    print(f"  融合权重: BM25={debug_info['config']['alpha']}, SimHash={debug_info['config']['beta']}")
    print(f"  查询SimHash: {debug_info['query_simhash']:016x}")
    
    # 显示SimHash相似度分布
    simhash_sims = debug_info['simhash_similarities']
    print(f"  SimHash相似度范围: {min(simhash_sims):.3f} - {max(simhash_sims):.3f}")
    print(f"  平均SimHash相似度: {np.mean(simhash_sims):.3f}")
    
    # 8. 保存详细结果
    result_summary = {
        'query_id': query_id,
        'original_ranking': candidate_ids,
        'reranked_ranking': reranked_ids,
        'metrics': metrics,
        'debug_info': debug_info,
        'config': config.__dict__
    }
    
    with open('reranking_result.json', 'w', encoding='utf-8') as f:
        json.dump(result_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细结果已保存到 reranking_result.json")
    print("🎊 演示完成!")


if __name__ == "__main__":
    main() 