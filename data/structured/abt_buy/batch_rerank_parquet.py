#!/usr/bin/env python3
"""
批量重排序脚本 - 处理完整的parquet数据集

输入文件:
- sparkly_results_k50.parquet: Sparkly的完整搜索结果
- table_a.parquet: 被索引的完整数据
- table_b.parquet: 查询的完整数据
- optimization_result.json: 字段权重配置
- gold.parquet: 真实标签(可选，用于评估)

输出文件:
- reranked_results_k50.parquet: 重排序后的完整结果
- rerank_evaluation_report.json: 性能评估报告

使用方法:
python batch_rerank_parquet.py [--config_file optimization_result.json] [--output_file reranked_results_k50.parquet]
"""

import pandas as pd
import numpy as np
import json
import time
import argparse
from typing import Dict, List, Tuple, Optional
from simhash_reranker import SimHashReranker, RerankerEvaluator, RerankerConfig


class ParquetBatchReranker:
    """完整parquet数据集的批量重排序器"""
    
    def __init__(self, config: RerankerConfig = None):
        self.config = config or RerankerConfig(
            simhash_bits=64,
            alpha=0.6,      # 使用最佳参数
            beta=0.4,
            use_3gram=True,
            normalize_scores=True
        )
        self.reranker = None
        self.evaluator = RerankerEvaluator()
        
    def load_data(self):
        """加载所有parquet数据"""
        print("📁 加载完整数据集...")
        
        # 加载表格数据
        self.table_a = pd.read_parquet('table_a.parquet')
        self.table_b = pd.read_parquet('table_b.parquet')
        
        # 加载Sparkly搜索结果
        self.search_results = pd.read_parquet('sparkly_results_k50.parquet')
        
        # 尝试加载真实标签(用于评估)
        try:
            self.gold_df = pd.read_parquet('gold.parquet')
            self.gold_mapping = dict(zip(self.gold_df['id2'], self.gold_df['id1']))
            print(f"✅ 加载了 {len(self.gold_mapping)} 个真实标签")
        except FileNotFoundError:
            print("⚠️ 未找到gold.parquet文件，将跳过性能评估")
            self.gold_df = None
            self.gold_mapping = {}
        
        print(f"✅ Table A: {len(self.table_a)} 条记录")
        print(f"✅ Table B: {len(self.table_b)} 条记录")
        print(f"✅ 搜索结果: {len(self.search_results)} 个查询")
        
        # 创建ID到记录的快速映射
        self.table_a_dict = self.table_a.set_index('_id').to_dict('index')
        self.table_b_dict = self.table_b.set_index('_id').to_dict('index')
        
    def initialize_reranker(self, config_file: str = 'optimization_result.json'):
        """初始化重排序器"""
        print("⚙️ 初始化重排序器...")
        self.reranker = SimHashReranker(self.config)
        self.reranker.load_optimization_result(config_file)
        print("✅ 重排序器初始化完成")
        
    def process_single_query(self, row: pd.Series) -> Dict:
        """处理单个查询的重排序"""
        query_id = row['_id']
        candidate_ids = row['ids']
        original_scores = row['scores']
        
        # 获取查询记录
        if query_id not in self.table_b_dict:
            return {
                'query_id': query_id,
                'success': False,
                'error': f'Query ID {query_id} not found in table_b'
            }
            
        query_record = self.table_b_dict[query_id]
        
        # 获取候选记录
        candidate_records = []
        valid_candidates = []
        valid_scores = []
        
        for i, cand_id in enumerate(candidate_ids):
            if cand_id in self.table_a_dict:
                candidate_records.append(self.table_a_dict[cand_id])
                valid_candidates.append(cand_id)
                valid_scores.append(original_scores[i])
        
        if not candidate_records:
            return {
                'query_id': query_id,
                'success': False,
                'error': 'No valid candidates found'
            }
        
        # 执行重排序
        try:
            ranking_indices, fused_scores, debug_info = self.reranker.rerank_candidates(
                query_record, candidate_records, valid_scores
            )
            
            # 重新排列结果
            reranked_ids = [valid_candidates[i] for i in ranking_indices]
            reranked_scores = [fused_scores[i] for i in range(len(fused_scores))]
            
            return {
                'query_id': query_id,
                'success': True,
                'original_ids': valid_candidates,
                'original_scores': valid_scores,
                'reranked_ids': reranked_ids,
                'reranked_scores': reranked_scores,
                'processing_time': debug_info['processing_time']
            }
            
        except Exception as e:
            return {
                'query_id': query_id,
                'success': False,
                'error': str(e)
            }
    
    def batch_rerank(self) -> Tuple[pd.DataFrame, Dict]:
        """批量重排序所有查询"""
        print("🔄 开始批量重排序...")
        
        reranked_results = []
        evaluation_metrics = []
        total_time = 0
        success_count = 0
        
        for idx, row in self.search_results.iterrows():
            if idx % 100 == 0:
                print(f"进度: {idx}/{len(self.search_results)} ({idx/len(self.search_results)*100:.1f}%)")
            
            result = self.process_single_query(row)
            
            if result['success']:
                success_count += 1
                total_time += result['processing_time']
                
                # 构建重排序结果行
                reranked_row = {
                    '_id': result['query_id'],
                    'ids': result['reranked_ids'],
                    'scores': result['reranked_scores'],
                    'search_time': row['search_time'],  # 保持原始搜索时间
                    'rerank_time': result['processing_time']
                }
                reranked_results.append(reranked_row)
                
                # 如果有真实标签，计算评估指标
                if result['query_id'] in self.gold_mapping:
                    metrics = self.evaluator.calculate_metrics(
                        result['query_id'],
                        result['original_ids'],
                        result['reranked_ids'], 
                        self.gold_mapping
                    )
                    evaluation_metrics.append(metrics)
            else:
                print(f"⚠️ 查询 {result['query_id']} 处理失败: {result['error']}")
        
        # 创建结果DataFrame
        reranked_df = pd.DataFrame(reranked_results)
        
        # 生成评估报告
        evaluation_report = self._generate_evaluation_report(
            evaluation_metrics, success_count, total_time
        )
        
        print(f"\n✅ 批量重排序完成!")
        print(f"成功处理: {success_count}/{len(self.search_results)} 个查询")
        print(f"总处理时间: {total_time:.4f} 秒")
        print(f"平均处理时间: {total_time/success_count:.4f} 秒/查询")
        
        return reranked_df, evaluation_report
    
    def _generate_evaluation_report(self, metrics_list: List[Dict], 
                                  success_count: int, total_time: float) -> Dict:
        """生成评估报告"""
        if not metrics_list:
            return {
                'total_queries_processed': success_count,
                'total_processing_time': total_time,
                'avg_processing_time': total_time / success_count if success_count > 0 else 0,
                'evaluation_available': False,
                'message': '没有可用的真实标签进行性能评估'
            }
        
        # 汇总评估指标
        report = {
            'total_queries_processed': success_count,
            'queries_with_ground_truth': len(metrics_list),
            'total_processing_time': total_time,
            'avg_processing_time': total_time / success_count if success_count > 0 else 0,
            'evaluation_available': True,
            'config': self.config.__dict__
        }
        
        # 计算各项指标的平均值
        for k in [1, 5, 10, 20]:
            if f'original_recall@{k}' in metrics_list[0]:
                orig_recalls = [m[f'original_recall@{k}'] for m in metrics_list]
                rerank_recalls = [m[f'reranked_recall@{k}'] for m in metrics_list]
                rank_improvements = [m[f'rank_improvement@{k}'] for m in metrics_list]
                
                report[f'avg_original_recall@{k}'] = np.mean(orig_recalls)
                report[f'avg_reranked_recall@{k}'] = np.mean(rerank_recalls)
                report[f'recall_improvement@{k}'] = np.mean(rerank_recalls) - np.mean(orig_recalls)
                report[f'avg_rank_improvement@{k}'] = np.mean(rank_improvements)
                report[f'positive_improvements@{k}'] = sum(1 for x in rank_improvements if x > 0)
                report[f'negative_improvements@{k}'] = sum(1 for x in rank_improvements if x < 0)
        
        return report
    
    def save_results(self, reranked_df: pd.DataFrame, 
                    evaluation_report: Dict, output_file: str):
        """保存重排序结果和评估报告"""
        # 保存重排序后的parquet文件
        reranked_df.to_parquet(output_file)
        print(f"💾 重排序结果已保存到: {output_file}")
        
        # 保存评估报告
        report_file = output_file.replace('.parquet', '_evaluation_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_report, f, indent=2, ensure_ascii=False)
        print(f"📊 评估报告已保存到: {report_file}")
        
        # 打印性能摘要
        if evaluation_report['evaluation_available']:
            print(f"\n📈 性能改善摘要:")
            for k in [1, 5, 10]:
                improvement = evaluation_report[f'recall_improvement@{k}']
                positive = evaluation_report[f'positive_improvements@{k}']
                total = evaluation_report['queries_with_ground_truth']
                print(f"   Recall@{k}: {improvement:+.4f} ({positive}/{total} 查询改善)")


def main():
    parser = argparse.ArgumentParser(description='批量重排序完整parquet数据集')
    parser.add_argument('--config_file', type=str, default='optimization_result.json',
                       help='字段权重配置文件路径')
    parser.add_argument('--output_file', type=str, default='reranked_results_k50.parquet',
                       help='输出文件路径')
    parser.add_argument('--alpha', type=float, default=0.6,
                       help='BM25权重 (默认: 0.6)')
    parser.add_argument('--beta', type=float, default=0.4,
                       help='SimHash权重 (默认: 0.4)')
    parser.add_argument('--simhash_bits', type=int, default=64,
                       help='SimHash位数 (默认: 64)')
    
    args = parser.parse_args()
    
    print("🚀 开始批量重排序完整parquet数据集")
    print("=" * 60)
    
    # 创建配置
    config = RerankerConfig(
        simhash_bits=args.simhash_bits,
        alpha=args.alpha,
        beta=args.beta,
        use_3gram=True,
        normalize_scores=True
    )
    
    print(f"🔧 配置参数: α={config.alpha}, β={config.beta}, bits={config.simhash_bits}")
    
    # 创建重排序器
    reranker = ParquetBatchReranker(config)
    
    # 加载数据
    reranker.load_data()
    
    # 初始化重排序器
    reranker.initialize_reranker(args.config_file)
    
    # 执行批量重排序
    reranked_df, evaluation_report = reranker.batch_rerank()
    
    # 保存结果
    reranker.save_results(reranked_df, evaluation_report, args.output_file)
    
    print("\n🎊 批量重排序完成!")
    print(f"📄 输出文件:")
    print(f"   - 重排序结果: {args.output_file}")
    print(f"   - 评估报告: {args.output_file.replace('.parquet', '_evaluation_report.json')}")


if __name__ == "__main__":
    main() 