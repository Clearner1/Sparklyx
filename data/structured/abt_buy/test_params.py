#!/usr/bin/env python3
"""
快速参数测试脚本 - 测试不同α、β组合对查询ID=2的重排序效果
"""

import json
import pandas as pd
from simhash_reranker import SimHashReranker, RerankerEvaluator, RerankerConfig

def test_parameter_combination(alpha, beta):
    """测试特定参数组合的效果"""
    
    # 加载数据
    table_a = pd.read_parquet('table_a.parquet')
    table_b = pd.read_parquet('table_b.parquet')
    
    with open('sparkly_results_k50.md', 'r', encoding='utf-8') as f:
        search_results = json.loads(f.read().strip())
    
    # 创建重排序器
    config = RerankerConfig(
        simhash_bits=64,
        alpha=alpha,
        beta=beta,
        use_3gram=True,
        normalize_scores=True
    )
    
    reranker = SimHashReranker(config)
    reranker.load_optimization_result('optimization_result.json')
    
    # 获取查询和候选数据
    query_id = search_results['_id']  # ID=2
    candidate_ids = search_results['ids']
    original_scores = search_results['scores']
    
    query_record = table_b[table_b['_id'] == query_id].iloc[0].to_dict()
    
    candidate_records = []
    for cand_id in candidate_ids:
        candidate_record = table_a[table_a['_id'] == cand_id].iloc[0].to_dict()
        candidate_records.append(candidate_record)
    
    # 执行重排序
    ranking_indices, fused_scores, debug_info = reranker.rerank_candidates(
        query_record, candidate_records, original_scores
    )
    
    # 分析结果
    reranked_ids = [candidate_ids[i] for i in ranking_indices]
    
    # 找到正确答案(ID=435)的排名
    true_answer_id = 435
    original_rank = candidate_ids.index(true_answer_id) + 1
    reranked_rank = reranked_ids.index(true_answer_id) + 1
    
    improvement = original_rank - reranked_rank
    
    return {
        'alpha': alpha,
        'beta': beta,
        'original_rank': original_rank,
        'reranked_rank': reranked_rank,
        'improvement': improvement,
        'top3_ids': reranked_ids[:3],
        'top3_scores': fused_scores[:3]
    }

def main():
    print("🔧 快速参数测试 - 查询ID=2案例")
    print("="*50)
    
    # 测试不同的参数组合
    param_combinations = [
        (0.9, 0.1),  # 重视BM25
        (0.8, 0.2),
        (0.7, 0.3),  # 当前默认
        (0.6, 0.4),
        (0.5, 0.5),  # 平衡
        (0.4, 0.6),  # 重视SimHash
        (0.3, 0.7),
        (0.2, 0.8),
    ]
    
    results = []
    
    for alpha, beta in param_combinations:
        print(f"测试参数: α={alpha}, β={beta}...")
        try:
            result = test_parameter_combination(alpha, beta)
            results.append(result)
            
            improvement_text = ""
            if result['improvement'] > 0:
                improvement_text = f"🎉 提升 {result['improvement']} 位"
            elif result['improvement'] < 0:
                improvement_text = f"⬇️ 下降 {abs(result['improvement'])} 位"
            else:
                improvement_text = "➡️ 无变化"
            
            print(f"  排名: {result['original_rank']} → {result['reranked_rank']} {improvement_text}")
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
    
    # 显示汇总结果
    print(f"\n📊 参数测试汇总 (正确答案ID=435):")
    print("-"*60)
    print(f"{'α':>5} {'β':>5} {'原排名':>8} {'新排名':>8} {'改善':>6} {'前3名':>15}")
    print("-"*60)
    
    best_improvement = float('-inf')
    best_params = None
    
    for result in results:
        improvement_str = f"+{result['improvement']}" if result['improvement'] > 0 else str(result['improvement'])
        top3_str = f"{result['top3_ids'][:3]}"
        
        print(f"{result['alpha']:>5.1f} {result['beta']:>5.1f} {result['original_rank']:>8d} "
              f"{result['reranked_rank']:>8d} {improvement_str:>6s} {top3_str}")
        
        if result['improvement'] > best_improvement:
            best_improvement = result['improvement']
            best_params = (result['alpha'], result['beta'])
    
    print("-"*60)
    
    if best_improvement > 0:
        print(f"🏆 最佳参数: α={best_params[0]}, β={best_params[1]} (提升{best_improvement}位)")
    else:
        print("⚠️ 当前测试中没有参数组合能提升排名")
        print("💡 建议: 1) 尝试更多参数组合 2) 调整SimHash算法 3) 检查字段权重")

if __name__ == "__main__":
    main() 