#!/usr/bin/env python3
"""
批量评估脚本 - 对整个abt_buy数据集进行SimHash重排序评估

功能:
1. 批量处理所有查询
2. 统计不同参数下的性能表现
3. 生成详细的评估报告
4. 参数优化建议

运行方式: python batch_evaluation.py
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import time
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from simhash_reranker import SimHashReranker, RerankerEvaluator, RerankerConfig


class BatchEvaluator:
    """批量评估器"""
    
    def __init__(self):
        self.table_a = None
        self.table_b = None
        self.gold_mapping = None
        self.evaluator = RerankerEvaluator()
        
    def load_data(self):
        """加载所有数据"""
        print("📁 加载数据集...")
        
        # 加载表格数据
        self.table_a = self._load_table('table_a.md')
        self.table_b = self._load_table('table_b.md')
        
        # 加载真实标签
        self.gold_mapping = self.evaluator.load_ground_truth('gold_part.md')
        
        print(f"✅ Table A: {len(self.table_a)} 条记录")
        print(f"✅ Table B: {len(self.table_b)} 条记录")
        print(f"✅ 真实标签: {len(self.gold_mapping)} 对")
        
    def _load_table(self, table_path: str) -> pd.DataFrame:
        """加载表格数据"""
        records = []
        with open(table_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))
        return pd.DataFrame(records)
    
    def simulate_search_results(self, k: int = 50) -> Dict[int, Dict]:
        """模拟搜索结果 - 为所有有真实标签的查询生成候选"""
        print(f"🔍 模拟搜索结果 (K={k})...")
        
        search_results = {}
        
        for query_id in self.gold_mapping.keys():
            if query_id >= len(self.table_b):
                continue
                
            # 获取查询记录
            query_record = self.table_b.iloc[query_id]
            query_text = f"{query_record['name']} {query_record['description']}".lower()
            
            # 简单的基于词汇重叠的相似度计算 (模拟BM25)
            candidate_scores = []
            
            for _, candidate in self.table_a.iterrows():
                candidate_text = f"{candidate['name']} {candidate['description']}".lower()
                
                # 计算简单的词汇重叠分数
                query_words = set(query_text.split())
                candidate_words = set(candidate_text.split())
                
                if len(query_words) == 0:
                    score = 0.0
                else:
                    overlap = len(query_words.intersection(candidate_words))
                    score = overlap / len(query_words) + np.random.normal(0, 0.1)  # 添加噪音
                
                candidate_scores.append((candidate['_id'], score))
            
            # 排序并取前K个
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            top_k = candidate_scores[:k]
            
            search_results[query_id] = {
                'ids': [item[0] for item in top_k],
                'scores': [item[1] for item in top_k]
            }
        
        return search_results
    
    def evaluate_configuration(self, 
                             config: RerankerConfig, 
                             search_results: Dict[int, Dict]) -> Dict:
        """评估特定配置的性能"""
        
        # 创建重排序器
        reranker = SimHashReranker(config)
        reranker.load_optimization_result('optimization_result.json')
        
        # 收集所有查询的评估结果
        all_metrics = []
        total_processing_time = 0
        
        for query_id, result in search_results.items():
            if query_id not in self.gold_mapping:
                continue
                
            # 获取查询和候选记录
            query_record = self.table_b.iloc[query_id].to_dict()
            
            candidate_records = []
            for cand_id in result['ids']:
                candidate_record = self.table_a[self.table_a['_id'] == cand_id].iloc[0].to_dict()
                candidate_records.append(candidate_record)
            
            # 执行重排序
            start_time = time.time()
            ranking_indices, fused_scores, debug_info = reranker.rerank_candidates(
                query_record, candidate_records, result['scores']
            )
            total_processing_time += time.time() - start_time
            
            # 重新排列候选ID
            reranked_ids = [result['ids'][i] for i in ranking_indices]
            
            # 评估效果
            metrics = self.evaluator.calculate_metrics(
                query_id, result['ids'], reranked_ids, self.gold_mapping
            )
            
            if metrics['has_ground_truth']:
                all_metrics.append(metrics)
        
        # 汇总统计
        summary = self._aggregate_metrics(all_metrics)
        summary['config'] = config.__dict__
        summary['total_queries'] = len(all_metrics)
        summary['avg_processing_time'] = total_processing_time / len(all_metrics) if all_metrics else 0
        
        return summary
    
    def _aggregate_metrics(self, all_metrics: List[Dict]) -> Dict:
        """汇总评估指标"""
        if not all_metrics:
            return {}
        
        summary = {}
        
        # 计算各个K值下的平均指标
        for k in [1, 5, 10, 20]:
            # Recall@K
            original_recalls = [m[f'original_recall@{k}'] for m in all_metrics]
            reranked_recalls = [m[f'reranked_recall@{k}'] for m in all_metrics]
            
            summary[f'avg_original_recall@{k}'] = np.mean(original_recalls)
            summary[f'avg_reranked_recall@{k}'] = np.mean(reranked_recalls)
            summary[f'recall_improvement@{k}'] = np.mean(reranked_recalls) - np.mean(original_recalls)
            
            # 排名改善
            rank_improvements = [m[f'rank_improvement@{k}'] for m in all_metrics]
            summary[f'avg_rank_improvement@{k}'] = np.mean(rank_improvements)
            summary[f'positive_rank_improvements@{k}'] = sum(1 for x in rank_improvements if x > 0)
            summary[f'negative_rank_improvements@{k}'] = sum(1 for x in rank_improvements if x < 0)
        
        return summary
    
    def grid_search_parameters(self, search_results: Dict[int, Dict]) -> pd.DataFrame:
        """网格搜索最佳参数"""
        print("🔧 网格搜索最佳参数...")
        
        # 参数网格
        alpha_values = [0.5, 0.6, 0.7, 0.8, 0.9]
        beta_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        simhash_bits = [32, 64]
        
        results = []
        total_combinations = len(alpha_values) * len(beta_values) * len(simhash_bits)
        
        print(f"总共需要测试 {total_combinations} 种参数组合...")
        
        for i, (alpha, beta, bits) in enumerate(product(alpha_values, beta_values, simhash_bits)):
            # 确保alpha + beta <= 1.0
            if alpha + beta > 1.0:
                continue
                
            print(f"进度: {i+1}/{total_combinations} - alpha={alpha}, beta={beta}, bits={bits}")
            
            config = RerankerConfig(
                simhash_bits=bits,
                alpha=alpha,
                beta=beta,
                use_3gram=True,
                normalize_scores=True
            )
            
            try:
                summary = self.evaluate_configuration(config, search_results)
                summary['alpha'] = alpha
                summary['beta'] = beta
                summary['simhash_bits'] = bits
                results.append(summary)
            except Exception as e:
                print(f"参数组合失败: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def generate_report(self, results_df: pd.DataFrame):
        """生成评估报告"""
        print("📊 生成评估报告...")
        
        # 保存详细结果
        results_df.to_csv('parameter_optimization_results.csv', index=False)
        
        # 找到最佳配置
        best_config = results_df.loc[results_df['recall_improvement@10'].idxmax()]
        
        print("\n" + "="*60)
        print("🏆 最佳配置:")
        print(f"   Alpha (BM25权重): {best_config['alpha']}")
        print(f"   Beta (SimHash权重): {best_config['beta']}")
        print(f"   SimHash位数: {best_config['simhash_bits']}")
        print(f"   Recall@10提升: {best_config['recall_improvement@10']:.4f}")
        print(f"   平均处理时间: {best_config['avg_processing_time']:.4f} 秒")
        
        # 显示不同K值下的改善情况
        print("\n📈 最佳配置下的性能提升:")
        for k in [1, 5, 10, 20]:
            improvement = best_config[f'recall_improvement@{k}']
            positive = best_config[f'positive_rank_improvements@{k}']
            negative = best_config[f'negative_rank_improvements@{k}']
            total = best_config['total_queries']
            
            print(f"   Recall@{k}: +{improvement:.4f} ({positive}/{total} 查询改善, {negative}/{total} 查询下降)")
        
        # 生成可视化报告
        self._create_visualizations(results_df)
    
    def _create_visualizations(self, results_df: pd.DataFrame):
        """创建可视化图表"""
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Alpha vs Recall@10 改善
        pivot_alpha = results_df.pivot_table(
            values='recall_improvement@10', 
            index='alpha', 
            columns='beta', 
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_alpha, annot=True, fmt='.4f', cmap='RdYlGn', 
                   center=0, ax=axes[0,0])
        axes[0,0].set_title('Recall@10改善 vs Alpha/Beta参数')
        
        # 2. SimHash位数对比
        bit_comparison = results_df.groupby('simhash_bits').agg({
            'recall_improvement@10': 'mean',
            'avg_processing_time': 'mean'
        }).reset_index()
        
        axes[0,1].bar(bit_comparison['simhash_bits'], bit_comparison['recall_improvement@10'])
        axes[0,1].set_title('不同SimHash位数的Recall@10改善')
        axes[0,1].set_xlabel('SimHash位数')
        axes[0,1].set_ylabel('平均Recall@10改善')
        
        # 3. 处理时间 vs 性能
        axes[1,0].scatter(results_df['avg_processing_time'], results_df['recall_improvement@10'],
                         alpha=0.6)
        axes[1,0].set_xlabel('平均处理时间 (秒)')
        axes[1,0].set_ylabel('Recall@10改善')
        axes[1,0].set_title('处理时间 vs 性能改善')
        
        # 4. 不同K值下的改善分布
        k_improvements = []
        for k in [1, 5, 10, 20]:
            k_improvements.extend([(k, x) for x in results_df[f'recall_improvement@{k}']])
        
        k_df = pd.DataFrame(k_improvements, columns=['K', 'Improvement'])
        sns.boxplot(data=k_df, x='K', y='Improvement', ax=axes[1,1])
        axes[1,1].set_title('不同K值下的Recall改善分布')
        axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('parameter_optimization_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 可视化报告已保存到 parameter_optimization_analysis.png")


def main():
    print("🚀 开始批量评估SimHash重排序")
    print("="*60)
    
    # 1. 初始化评估器
    evaluator = BatchEvaluator()
    evaluator.load_data()
    
    # 2. 模拟搜索结果 (在实际应用中，这些来自Sparkly的搜索结果)
    search_results = evaluator.simulate_search_results(k=50)
    print(f"✅ 生成了 {len(search_results)} 个查询的搜索结果")
    
    # 3. 网格搜索最佳参数
    results_df = evaluator.grid_search_parameters(search_results)
    
    # 4. 生成报告
    evaluator.generate_report(results_df)
    
    print("\n🎊 批量评估完成!")
    print("📄 检查以下文件获取详细结果:")
    print("   - parameter_optimization_results.csv")
    print("   - parameter_optimization_analysis.png")


if __name__ == "__main__":
    main() 