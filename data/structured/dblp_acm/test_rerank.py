#!/usr/bin/env python3
"""
通用数据集重排序详细评估脚本

功能:
1. 批量处理任意数据集的所有查询
2. 生成详细的重排序效果评估报告
3. 对比原始BM25 vs SimHash重排序的性能
4. 输出可视化的结果分析

输出文件:
- {dataset}_reranked_results_k50.parquet: 重排序结果
- {dataset}_evaluation_report.json: 详细评估报告
- {dataset}_sample_cases.json: 典型案例分析

使用方法:
cd data/structured/{any_dataset}/
python test_rerank_ag.py [--alpha 0.6] [--beta 0.4] [--sample_size 100]
"""

import json
import pandas as pd
import numpy as np
import sys
import os
import time
import argparse
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# 添加 abt_buy 目录到 Python 路径，以便导入 simhash_reranker
sys.path.append('../abt_buy')
from simhash_reranker import SimHashReranker, RerankerEvaluator, RerankerConfig


class UniversalDatasetEvaluator:
    """通用数据集评估器"""
    
    def __init__(self, config: RerankerConfig = None):
        self.config = config or RerankerConfig(
            simhash_bits=64,
            alpha=0.6,
            beta=0.4,
            use_3gram=True,
            normalize_scores=True
        )
        # 自动检测数据集名称
        self.dataset_name = Path.cwd().name
        
    def load_data(self):
        """加载数据集"""
        print(f"📁 加载 {self.dataset_name} 数据集...")
        
        # 检查必要文件
        required_files = [
            'table_a.parquet', 'table_b.parquet', 
            'sparkly_results_k50.parquet', 'optimization_result.json'
        ]
        
        missing_files = [f for f in required_files if not Path(f).exists()]
        if missing_files:
            raise FileNotFoundError(f"缺少必要文件: {missing_files}")

# 加载数据
        self.table_a = pd.read_parquet('table_a.parquet')
        self.table_b = pd.read_parquet('table_b.parquet')
        self.search_results = pd.read_parquet('sparkly_results_k50.parquet')
        
        # 加载真实标签
        try:
            gold_df = pd.read_parquet('gold.parquet')
            self.gold_mapping = dict(zip(gold_df['id2'], gold_df['id1']))
            print(f"✅ 加载了 {len(self.gold_mapping)} 个真实标签")
        except FileNotFoundError:
            print("⚠️ 未找到gold.parquet文件，将无法进行准确性评估")
            self.gold_mapping = {}
        
        print(f"✅ Table A: {len(self.table_a)} 条记录")
        print(f"✅ Table B: {len(self.table_b)} 条记录") 
        print(f"✅ 搜索结果: {len(self.search_results)} 个查询")
        
        # 创建快速映射
        self.table_a_dict = self.table_a.set_index('_id').to_dict('index')
        self.table_b_dict = self.table_b.set_index('_id').to_dict('index')
        
    def initialize_reranker(self):
        """初始化重排序器"""
        print("⚙️ 初始化SimHash重排序器...")
        
        self.reranker = SimHashReranker(self.config)
        self.reranker.load_optimization_result('optimization_result.json')
        self.evaluator = RerankerEvaluator()
        
        print(f"🔧 配置参数: α={self.config.alpha} (BM25), β={self.config.beta} (SimHash)")
    
    def process_sample_queries(self, sample_size: int = 100):
        """处理样本查询，用于快速测试"""
        print(f"\n🎯 处理样本查询 (前{sample_size}个)...")
        
        # 如果有真实标签，优先选择有标签的查询
        if self.gold_mapping:
            labeled_queries = self.search_results[self.search_results['_id'].isin(self.gold_mapping.keys())]
            if len(labeled_queries) >= sample_size:
                sample_results = labeled_queries.head(sample_size)
                print(f"✅ 选择了 {sample_size} 个有真实标签的查询")
            else:
                # 先取所有有标签的，再补充无标签的
                unlabeled_queries = self.search_results[~self.search_results['_id'].isin(self.gold_mapping.keys())]
                remaining = sample_size - len(labeled_queries)
                sample_results = pd.concat([labeled_queries, unlabeled_queries.head(remaining)])
                print(f"✅ 选择了 {len(labeled_queries)} 个有标签查询 + {remaining} 个无标签查询")
        else:
            sample_results = self.search_results.head(sample_size)
            print(f"⚠️ 没有真实标签，选择前 {sample_size} 个查询")
        success_count = 0
        total_time = 0
        sample_cases = []
        reranked_results = []
        evaluation_metrics = []
        
        for idx, row in sample_results.iterrows():
            result = self._process_single_query(row)
            
            if result['success']:
                success_count += 1
                total_time += result['processing_time']
                
                # 显示前5个详细案例
                if idx < 5:
                    self._show_query_details(row, result)
                
                # 构建重排序结果
                reranked_row = {
                    '_id': int(result['query_id']),
                    'ids': [int(x) for x in result['reranked_ids']],
                    'scores': [float(x) for x in result['reranked_scores']],
                    'search_time': float(row.get('search_time', 0)),
                    'rerank_time': float(result['processing_time'])
                }
                reranked_results.append(reranked_row)
                
                # 收集评估指标
                if result['metrics'] and result['metrics']['has_ground_truth']:
                    evaluation_metrics.append(result['metrics'])
                    # 收集有趣案例
                    if any(result['metrics'][f'rank_improvement@{k}'] != 0 for k in [1, 5, 10]):
                        case = self._create_simple_case(row, result)
                        sample_cases.append(case)
        
        print(f"\n✅ 样本处理完成: {success_count}/{len(sample_results)} 查询成功")
        print(f"⏱️ 平均处理时间: {total_time/success_count*1000:.2f} ms")
        
        # 调试信息
        print(f"🔍 调试信息:")
        print(f"   有真实标签的查询数: {len(evaluation_metrics)}")
        print(f"   有趣案例数: {len(sample_cases)}")
        
        # 保存样本结果
        if reranked_results:
            reranked_df = pd.DataFrame(reranked_results)
            output_file = f"{self.dataset_name}_sample_reranked_results.parquet"
            reranked_df.to_parquet(output_file)
            print(f"💾 样本重排序结果已保存: {output_file}")
        
        # 生成样本评估报告
        if evaluation_metrics:
            print(f"📊 生成样本评估报告...")
            evaluation_report = self._generate_evaluation_report(
                evaluation_metrics, success_count, total_time
            )
            # 标记为样本模式
            evaluation_report['sample_mode'] = True
            evaluation_report['sample_size'] = sample_size
            
            report_file = f"{self.dataset_name}_sample_evaluation_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_report, f, indent=2, ensure_ascii=False)
            print(f"📊 样本评估报告已保存: {report_file}")
            
            # 打印样本摘要
            self.print_sample_summary(evaluation_report)
        
        else:
            print(f"⚠️ 样本中没有有真实标签的查询，无法生成评估报告")
        
        # 保存有趣案例
        if sample_cases:
            print(f"📝 保存有趣案例...")
            cases_file = f"{self.dataset_name}_sample_cases.json"
            with open(cases_file, 'w', encoding='utf-8') as f:
                json.dump(sample_cases, f, indent=2, ensure_ascii=False, default=str)
            print(f"📝 有趣案例已保存: {cases_file}")
        else:
            print(f"⚠️ 样本中没有有趣案例（排名变化），无法生成案例文件")
        
        return sample_cases
    
    def process_all_queries(self):
        """处理所有查询"""
        print(f"\n🚀 开始批量处理所有 {len(self.search_results)} 个查询...")
        
        reranked_results = []
        evaluation_metrics = []
        total_rerank_time = 0
        success_count = 0
        
        for idx, row in self.search_results.iterrows():
            # 显示进度
            if idx % max(1, len(self.search_results) // 20) == 0:
                progress = idx / len(self.search_results) * 100
                print(f"  进度: {idx}/{len(self.search_results)} ({progress:.1f}%)")
            
            # 处理查询
            result = self._process_single_query(row)
            
            if result['success']:
                success_count += 1
                total_rerank_time += result['processing_time']
                
                # 构建重排序结果
                reranked_row = {
                    '_id': int(result['query_id']),
                    'ids': [int(x) for x in result['reranked_ids']],
                    'scores': [float(x) for x in result['reranked_scores']],
                    'search_time': float(row.get('search_time', 0)),
                    'rerank_time': float(result['processing_time'])
                }
                reranked_results.append(reranked_row)
                
                # 评估指标
                if result['metrics'] and result['metrics']['has_ground_truth']:
                    evaluation_metrics.append(result['metrics'])
        
        print(f"✅ 处理完成: {success_count}/{len(self.search_results)} 查询成功")
        
        # 保存重排序结果
        if reranked_results:
            reranked_df = pd.DataFrame(reranked_results)
            output_file = f"{self.dataset_name}_reranked_results_k50.parquet"
            reranked_df.to_parquet(output_file)
            print(f"💾 重排序结果已保存: {output_file}")
        
        # 生成评估报告
        evaluation_report = self._generate_evaluation_report(
            evaluation_metrics, success_count, total_rerank_time
        )
        
        # 保存评估报告
        report_file = f"{self.dataset_name}_evaluation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_report, f, indent=2, ensure_ascii=False)
        print(f"📊 评估报告已保存: {report_file}")
        
        return evaluation_report
    
    def _process_single_query(self, row: pd.Series) -> Dict:
        """处理单个查询"""
        query_id = row['_id']
        candidate_ids = row['ids']
        original_scores = row['scores']
        
        # 获取查询记录
        if query_id not in self.table_b_dict:
            return {'success': False, 'error': f'Query ID {query_id} not found'}
            
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
            return {'success': False, 'error': 'No valid candidates'}
        
        # 执行重排序
        try:
            ranking_indices, fused_scores, debug_info = self.reranker.rerank_candidates(
                query_record, candidate_records, valid_scores
            )
            
            reranked_ids = [valid_candidates[i] for i in ranking_indices]
            
            # 评估指标
            metrics = None
            if query_id in self.gold_mapping:
                metrics = self.evaluator.calculate_metrics(
                    query_id, valid_candidates, reranked_ids, self.gold_mapping
                )
            
            return {
                'success': True,
                'query_id': query_id,
                'reranked_ids': reranked_ids,
                'reranked_scores': fused_scores,
                'processing_time': debug_info['processing_time'],
                'metrics': metrics
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _show_query_details(self, row: pd.Series, result: Dict):
        """显示查询详细信息"""
        query_id = result['query_id']
        query_record = self.table_b_dict[query_id]
        
        print(f"\n🔍 查询案例 (ID={query_id}):")
        
        # 动态显示所有字段
        for key, value in query_record.items():
            if key != '_id' and value and str(value) != 'nan':
                print(f"  {key}: {value}")
        
        if result['metrics'] and result['metrics']['has_ground_truth']:
            metrics = result['metrics']
            true_id = metrics['true_match_id']
            print(f"  ✅ 真实匹配: ID={true_id}")
            
            for k in [1, 5, 10]:
                orig_rank = metrics[f'original_rank@{k}']
                rerank_rank = metrics[f'reranked_rank@{k}']
                improvement = metrics[f'rank_improvement@{k}']
                
                if orig_rank > 0 and rerank_rank > 0:
                    if improvement > 0:
                        print(f"  📈 Rank@{k}: {orig_rank} → {rerank_rank} (+{improvement})")
                    elif improvement < 0:
                        print(f"  📉 Rank@{k}: {orig_rank} → {rerank_rank} ({improvement})")
                    else:
                        print(f"  ➡️ Rank@{k}: {orig_rank} (无变化)")
        
        print(f"  ⏱️ 重排序时间: {result['processing_time']:.4f}s")
    
    def _create_simple_case(self, row: pd.Series, result: Dict) -> Dict:
        """创建简化案例"""
        query_id = result['query_id']
        query_record = self.table_b_dict[query_id]
        
        # 转换metrics中的numpy类型为Python原生类型
        metrics = result['metrics']
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, np.integer):
                clean_metrics[k] = int(v)
            elif isinstance(v, np.floating):
                clean_metrics[k] = float(v)
            elif isinstance(v, (int, float)):
                clean_metrics[k] = v
            else:
                clean_metrics[k] = v
        
        case = {
            'query_id': int(query_id),
            'query_info': {k: str(v) for k, v in query_record.items() if k != '_id'},
            'metrics': clean_metrics,
            'processing_time': float(result['processing_time'])
        }
        
        return case
    
    def _generate_evaluation_report(self, metrics_list: List[Dict], 
                                  success_count: int, total_rerank_time: float) -> Dict:
        """生成评估报告"""
        
        report = {
            'dataset': self.dataset_name,
            'total_queries': len(self.search_results),
            'successful_queries': success_count,
            'queries_with_ground_truth': len(metrics_list),
            'total_rerank_time': total_rerank_time,
            'avg_rerank_time': total_rerank_time / success_count if success_count > 0 else 0,
            'config': self.config.__dict__,
            'evaluation_available': len(metrics_list) > 0
        }
        
        if metrics_list:
            # 计算各项指标
            for k in [1, 5, 10, 20]:
                orig_recalls = [m[f'original_recall@{k}'] for m in metrics_list]
                rerank_recalls = [m[f'reranked_recall@{k}'] for m in metrics_list]
                rank_improvements = [m[f'rank_improvement@{k}'] for m in metrics_list]
                
                report[f'avg_original_recall@{k}'] = float(np.mean(orig_recalls))
                report[f'avg_reranked_recall@{k}'] = float(np.mean(rerank_recalls))
                report[f'recall_improvement@{k}'] = float(np.mean(rerank_recalls) - np.mean(orig_recalls))
                report[f'avg_rank_improvement@{k}'] = float(np.mean(rank_improvements))
                report[f'positive_improvements@{k}'] = int(sum(1 for x in rank_improvements if x > 0))
                report[f'negative_improvements@{k}'] = int(sum(1 for x in rank_improvements if x < 0))
        
        return report
    
    def print_summary(self, evaluation_report: Dict):
        """打印评估摘要"""
        print(f"\n{'='*60}")
        print(f"🏆 {self.dataset_name} 数据集重排序评估摘要")
        print(f"{'='*60}")
        
        print(f"📊 处理统计:")
        print(f"   总查询数: {evaluation_report['total_queries']}")
        print(f"   成功处理: {evaluation_report['successful_queries']}")
        print(f"   有真实标签: {evaluation_report['queries_with_ground_truth']}")
        print(f"   平均重排序时间: {evaluation_report['avg_rerank_time']*1000:.2f} ms")
        
        if evaluation_report['evaluation_available']:
            print(f"\n📈 性能改善:")
            for k in [1, 5, 10]:
                improvement = evaluation_report[f'recall_improvement@{k}']
                positive = evaluation_report[f'positive_improvements@{k}']
                negative = evaluation_report[f'negative_improvements@{k}']
                total = evaluation_report['queries_with_ground_truth']
                
                status = "🟢" if improvement > 0 else "🔴" if improvement < 0 else "🟡"
                print(f"   {status} Recall@{k}: {improvement:+.4f} ({positive}↑/{negative}↓/{total} 查询)")
        
        print(f"\n💾 输出文件:")
        print(f"   📄 {self.dataset_name}_reranked_results_k50.parquet")
        print(f"   📊 {self.dataset_name}_evaluation_report.json")
    
    def print_sample_summary(self, evaluation_report: Dict):
        """打印样本评估摘要"""
        print(f"\n{'='*50}")
        print(f"🏆 {self.dataset_name} 样本评估摘要")
        print(f"{'='*50}")
        
        print(f"📊 样本统计:")
        print(f"   样本大小: {evaluation_report['sample_size']}")
        print(f"   成功处理: {evaluation_report['successful_queries']}")
        print(f"   有真实标签: {evaluation_report['queries_with_ground_truth']}")
        print(f"   平均重排序时间: {evaluation_report['avg_rerank_time']*1000:.2f} ms")
        
        if evaluation_report['evaluation_available']:
            print(f"\n📈 性能改善:")
            for k in [1, 5, 10]:
                improvement = evaluation_report[f'recall_improvement@{k}']
                positive = evaluation_report[f'positive_improvements@{k}']
                negative = evaluation_report[f'negative_improvements@{k}']
                total = evaluation_report['queries_with_ground_truth']
                
                status = "🟢" if improvement > 0 else "🔴" if improvement < 0 else "🟡"
                print(f"   {status} Recall@{k}: {improvement:+.4f} ({positive}↑/{negative}↓/{total} 查询)")
        
        print(f"\n💾 样本输出文件:")
        print(f"   📄 {self.dataset_name}_sample_reranked_results.parquet")
        print(f"   📊 {self.dataset_name}_sample_evaluation_report.json")
        print(f"   📝 {self.dataset_name}_sample_cases.json")


def main():
    parser = argparse.ArgumentParser(description='通用数据集重排序详细评估')
    parser.add_argument('--alpha', type=float, default=0.6,
                       help='BM25权重 (默认: 0.6)')
    parser.add_argument('--beta', type=float, default=0.4,
                       help='SimHash权重 (默认: 0.4)')
    parser.add_argument('--simhash_bits', type=int, default=64,
                       help='SimHash位数 (默认: 64)')
    parser.add_argument('--sample_only', action='store_true',
                       help='只处理样本查询，用于快速测试')
    parser.add_argument('--sample_size', type=int, default=100,
                       help='样本大小 (默认: 100)')
    
    args = parser.parse_args()
    
    # 自动检测数据集名称
    dataset_name = Path.cwd().name
    print(f"🌟 {dataset_name} 数据集SimHash重排序评估")
    print("=" * 60)
    
    # 创建配置
    config = RerankerConfig(
        simhash_bits=args.simhash_bits,
        alpha=args.alpha,
        beta=args.beta,
        use_3gram=True,
        normalize_scores=True
    )
    
    # 创建评估器
    evaluator = UniversalDatasetEvaluator(config)
    
    try:
        # 加载数据
        evaluator.load_data()

# 初始化重排序器
        evaluator.initialize_reranker()
        
        # 处理查询
        if args.sample_only:
            print(f"\n🎯 样本模式：处理前 {args.sample_size} 个查询")
            evaluator.process_sample_queries(args.sample_size)
        else:
            evaluation_report = evaluator.process_all_queries()
            evaluator.print_summary(evaluation_report)
        
    except Exception as e:
        print(f"❌ 评估过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()