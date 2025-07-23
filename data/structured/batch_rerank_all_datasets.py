#!/usr/bin/env python3
"""
批量重排序脚本 - 处理structured文件夹下所有数据集

功能:
1. 自动发现并处理所有数据集
2. 对每个数据集执行SimHash重排序
3. 生成详细的性能对比报告
4. 输出论文级别的结果汇总

输出文件:
- 每个数据集: {dataset}_reranked_results_k50.parquet
- 每个数据集: {dataset}_evaluation_report.json  
- 汇总报告: all_datasets_rerank_summary.json
- 汇总报告: all_datasets_rerank_summary.csv

使用方法:
cd data/structured/
python batch_rerank_all_datasets.py [--alpha 0.6] [--beta 0.4] [--sample_only]
"""

import os
import pandas as pd
import numpy as np
import json
import time
import argparse
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

# 导入重排序相关模块
sys.path.append('./abt_buy')  # 添加abt_buy路径
from simhash_reranker import SimHashReranker, RerankerEvaluator, RerankerConfig


class UniversalDatasetEvaluator:
    """通用数据集评估器（基于test_rerank.py）"""
    
    def __init__(self, config: RerankerConfig, dataset_path: Path):
        self.config = config
        self.dataset_name = dataset_path.name
        self.dataset_path = dataset_path
        
    def load_data(self):
        """加载数据集"""
        print(f"📁 加载 {self.dataset_name} 数据集...")
        
        # 检查必要文件
        required_files = [
            'table_a.parquet', 'table_b.parquet', 
            'sparkly_results_k50.parquet', 'optimization_result.json'
        ]
        
        missing_files = [f for f in required_files if not (self.dataset_path / f).exists()]
        if missing_files:
            raise FileNotFoundError(f"缺少必要文件: {missing_files}")

        # 加载数据
        self.table_a = pd.read_parquet(self.dataset_path / 'table_a.parquet')
        self.table_b = pd.read_parquet(self.dataset_path / 'table_b.parquet')
        self.search_results = pd.read_parquet(self.dataset_path / 'sparkly_results_k50.parquet')
        
        # 加载真实标签
        try:
            gold_df = pd.read_parquet(self.dataset_path / 'gold.parquet')
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
        self.reranker.load_optimization_result(str(self.dataset_path / 'optimization_result.json'))
        self.evaluator = RerankerEvaluator()
        
        print(f"🔧 配置参数: α={self.config.alpha} (BM25), β={self.config.beta} (SimHash)")
    
    def process_all_queries(self):
        """处理所有查询"""
        print(f"\n🚀 开始批量处理所有 {len(self.search_results)} 个查询...")
        
        reranked_results = []
        evaluation_metrics = []
        total_rerank_time = 0
        success_count = 0
        error_count = 0
        
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
                
                # 构建重排序结果（确保类型正确）
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
            else:
                error_count += 1
                if error_count <= 5:  # 只显示前5个错误
                    print(f"  ⚠️ 查询 {row['_id']} 处理失败: {result.get('error', 'Unknown error')}")
        
        print(f"✅ 处理完成: {success_count}/{len(self.search_results)} 查询成功")
        if error_count > 0:
            print(f"⚠️ {error_count} 个查询处理失败")
        
        return {
            'reranked_results': reranked_results,
            'evaluation_metrics': evaluation_metrics,
            'success_count': success_count,
            'total_rerank_time': total_rerank_time,
            'error_count': error_count
        }
    
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
    
    def save_results(self, results: Dict) -> Dict:
        """保存结果并生成报告"""
        # 保存重排序结果
        if results['reranked_results']:
            reranked_df = pd.DataFrame(results['reranked_results'])
            output_file = self.dataset_path / f"{self.dataset_name}_reranked_results_k50.parquet"
            reranked_df.to_parquet(output_file)
            print(f"💾 重排序结果已保存: {output_file}")
        
        # 生成评估报告
        evaluation_report = self._generate_evaluation_report(
            results['evaluation_metrics'], 
            results['success_count'], 
            results['total_rerank_time']
        )
        
        # 保存评估报告（确保JSON序列化安全）
        report_file = self.dataset_path / f"{self.dataset_name}_evaluation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_report, f, indent=2, ensure_ascii=False)
        print(f"📊 评估报告已保存: {report_file}")
        
        return evaluation_report
    
    def _generate_evaluation_report(self, metrics_list: List[Dict], 
                                  success_count: int, total_rerank_time: float) -> Dict:
        """生成评估报告"""
        
        report = {
            'dataset': self.dataset_name,
            'total_queries': len(self.search_results),
            'successful_queries': success_count,
            'queries_with_ground_truth': len(metrics_list),
            'total_rerank_time': float(total_rerank_time),
            'avg_rerank_time': float(total_rerank_time / success_count if success_count > 0 else 0),
            'config': self.config.__dict__,
            'evaluation_available': len(metrics_list) > 0
        }
        
        if metrics_list:
            # 计算各项指标（确保类型安全）
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


class MultiDatasetReranker:
    """多数据集批量重排序器（改进版）"""
    
    def __init__(self, config: RerankerConfig = None):
        self.config = config or RerankerConfig(
            simhash_bits=64,
            alpha=0.6,      # 使用论文中的最佳参数
            beta=0.4,
            use_3gram=True,
            normalize_scores=True
        )
        self.datasets = []
        self.results = {}
        
    def discover_datasets(self, base_dir: str = '.') -> List[Path]:
        """自动发现所有数据集"""
        print("🔍 发现可用数据集...")
        
        datasets = []
        base_path = Path(base_dir)
        
        for item in base_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # 检查必要的文件是否存在
                required_files = [
                    'sparkly_results_k50.parquet',
                    'table_a.parquet', 
                    'table_b.parquet',
                    'optimization_result.json'
                ]
                
                if all((item / f).exists() for f in required_files):
                    datasets.append(item)
                    print(f"  ✅ {item.name}")
                else:
                    missing = [f for f in required_files if not (item / f).exists()]
                    print(f"  ❌ {item.name} (缺少: {', '.join(missing)})")
        
        return sorted(datasets, key=lambda x: x.name)
    
    def process_single_dataset(self, dataset_path: Path) -> Dict:
        """处理单个数据集"""
        dataset_name = dataset_path.name
        print(f"\n{'='*60}")
        print(f"🚀 处理数据集: {dataset_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # 创建数据集评估器
            evaluator = UniversalDatasetEvaluator(self.config, dataset_path)
            
            # 加载数据
            evaluator.load_data()
            
            # 初始化重排序器
            evaluator.initialize_reranker()
            
            # 处理所有查询
            results = evaluator.process_all_queries()
            
            # 保存结果
            evaluation_report = evaluator.save_results(results)
            
            # 打印性能摘要
            if evaluation_report['evaluation_available']:
                print(f"\n📈 {dataset_name} 性能改善:")
                for k in [1, 5, 10]:
                    improvement = evaluation_report[f'recall_improvement@{k}']
                    positive = evaluation_report[f'positive_improvements@{k}']
                    total_eval = evaluation_report['queries_with_ground_truth']
                    print(f"   Recall@{k}: {improvement:+.4f} ({positive}/{total_eval} 查询改善)")
            else:
                print(f"⚠️ {dataset_name} 无评估数据（缺少真实标签）")
            
            # 返回汇总信息
            processing_time = time.time() - start_time
            
            return {
                'dataset': dataset_name,
                'success': True,
                'total_queries': len(evaluator.search_results),
                'successful_queries': results['success_count'],
                'queries_with_ground_truth': len(results['evaluation_metrics']),
                'processing_time': processing_time,
                'avg_rerank_time': results['total_rerank_time'] / results['success_count'] if results['success_count'] > 0 else 0,
                'evaluation_report': evaluation_report,
                'error_count': results['error_count']
            }
            
        except Exception as e:
            print(f"❌ 处理 {dataset_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'dataset': dataset_name,
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def process_all_datasets(self) -> Dict:
        """处理所有数据集"""
        print("🌟 开始批量处理所有数据集")
        print("=" * 80)
        
        # 发现数据集
        datasets = self.discover_datasets()
        
        if not datasets:
            print("❌ 未发现任何可处理的数据集!")
            return {}
        
        print(f"\n📊 发现 {len(datasets)} 个数据集: {', '.join([d.name for d in datasets])}")
        print(f"🔧 使用参数: α={self.config.alpha}, β={self.config.beta}, bits={self.config.simhash_bits}")
        
        # 处理每个数据集
        all_results = {}
        total_start_time = time.time()
        
        for i, dataset_path in enumerate(datasets):
            print(f"\n🎯 处理进度: {i+1}/{len(datasets)}")
            result = self.process_single_dataset(dataset_path)
            all_results[dataset_path.name] = result
        
        total_time = time.time() - total_start_time
        
        # 生成汇总报告
        summary = self._generate_summary_report(all_results, total_time)
        
        # 保存汇总报告
        with open('all_datasets_rerank_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 生成CSV汇总表
        self._generate_csv_summary(all_results)
        
        print(f"\n🎊 所有数据集处理完成! 总用时: {total_time:.2f} 秒")
        print("📄 输出文件:")
        print("   - all_datasets_rerank_summary.json (详细汇总)")
        print("   - all_datasets_rerank_summary.csv (表格汇总)")
        
        # 打印最终统计
        self._print_final_statistics(summary)
        
        return summary
    
    def _generate_summary_report(self, all_results: Dict, total_time: float) -> Dict:
        """生成汇总报告"""
        
        summary = {
            'experiment_config': self.config.__dict__,
            'total_processing_time': float(total_time),
            'datasets_processed': len(all_results),
            'successful_datasets': sum(1 for r in all_results.values() if r['success']),
            'datasets': {}
        }
        
        # 收集所有成功的数据集结果
        successful_datasets = []
        
        for dataset, result in all_results.items():
            summary['datasets'][dataset] = result
            
            if result['success'] and 'evaluation_report' in result:
                eval_report = result['evaluation_report']
                if eval_report['evaluation_available']:
                    successful_datasets.append(eval_report)
        
        # 计算整体统计
        if successful_datasets:
            summary['overall_statistics'] = {}
            
            total_queries = sum(d['queries_with_ground_truth'] for d in successful_datasets)
            summary['overall_statistics']['total_evaluated_queries'] = total_queries
            summary['overall_statistics']['datasets_with_evaluation'] = len(successful_datasets)
            
            # 加权平均各项指标
            for k in [1, 5, 10]:
                weighted_orig_recall = 0
                weighted_rerank_recall = 0
                total_positive_improvements = 0
                
                for d in successful_datasets:
                    if d['queries_with_ground_truth'] > 0:
                        weight = d['queries_with_ground_truth'] / total_queries
                        weighted_orig_recall += d[f'avg_original_recall@{k}'] * weight
                        weighted_rerank_recall += d[f'avg_reranked_recall@{k}'] * weight
                        total_positive_improvements += d[f'positive_improvements@{k}']
                
                summary['overall_statistics'][f'weighted_avg_original_recall@{k}'] = float(weighted_orig_recall)
                summary['overall_statistics'][f'weighted_avg_reranked_recall@{k}'] = float(weighted_rerank_recall)
                summary['overall_statistics'][f'overall_recall_improvement@{k}'] = float(weighted_rerank_recall - weighted_orig_recall)
                summary['overall_statistics'][f'total_positive_improvements@{k}'] = int(total_positive_improvements)
        
        return summary
    
    def _generate_csv_summary(self, all_results: Dict):
        """生成CSV格式的汇总表"""
        
        csv_data = []
        
        for dataset, result in all_results.items():
            if result['success'] and 'evaluation_report' in result:
                eval_report = result['evaluation_report']
                
                row = {
                    'Dataset': dataset,
                    'Success': result['success'],
                    'Total_Queries': eval_report['total_queries'],
                    'Successful_Queries': eval_report['successful_queries'],
                    'Evaluated_Queries': eval_report['queries_with_ground_truth'],
                    'Avg_Rerank_Time_ms': eval_report['avg_rerank_time'] * 1000,
                    'Processing_Time_s': result['processing_time'],
                    'Error_Count': result.get('error_count', 0)
                }
                
                # 添加Recall指标
                for k in [1, 5, 10]:
                    if eval_report['evaluation_available']:
                        row[f'Original_Recall@{k}'] = eval_report.get(f'avg_original_recall@{k}', 0)
                        row[f'Reranked_Recall@{k}'] = eval_report.get(f'avg_reranked_recall@{k}', 0)
                        row[f'Recall_Improvement@{k}'] = eval_report.get(f'recall_improvement@{k}', 0)
                        row[f'Positive_Improvements@{k}'] = eval_report.get(f'positive_improvements@{k}', 0)
                    else:
                        row[f'Original_Recall@{k}'] = 'N/A'
                        row[f'Reranked_Recall@{k}'] = 'N/A'
                        row[f'Recall_Improvement@{k}'] = 'N/A'
                        row[f'Positive_Improvements@{k}'] = 'N/A'
                
                csv_data.append(row)
            else:
                # 失败的数据集
                row = {
                    'Dataset': dataset,
                    'Success': False,
                    'Error': result.get('error', 'Unknown error'),
                    'Processing_Time_s': result.get('processing_time', 0)
                }
                csv_data.append(row)
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv('all_datasets_rerank_summary.csv', index=False)
            print("📊 CSV汇总表已保存: all_datasets_rerank_summary.csv")
    
    def _print_final_statistics(self, summary: Dict):
        """打印最终统计信息"""
        if 'overall_statistics' in summary:
            stats = summary['overall_statistics']
            print(f"\n🏆 整体性能统计:")
            print(f"   成功处理数据集: {summary['successful_datasets']}/{summary['datasets_processed']}")
            print(f"   有评估数据的数据集: {stats['datasets_with_evaluation']}")
            print(f"   总评估查询数: {stats['total_evaluated_queries']}")
            
            for k in [1, 5, 10]:
                improvement = stats[f'overall_recall_improvement@{k}']
                positive = stats[f'total_positive_improvements@{k}']
                total = stats['total_evaluated_queries']
                status = "🟢" if improvement > 0 else "🔴" if improvement < 0 else "🟡"
                print(f"   {status} Overall Recall@{k}: {improvement:+.4f} ({positive}/{total} 查询改善)")


def main():
    parser = argparse.ArgumentParser(description='批量重排序所有structured数据集')
    parser.add_argument('--alpha', type=float, default=0.6,
                       help='BM25权重 (默认: 0.6)')
    parser.add_argument('--beta', type=float, default=0.4,
                       help='SimHash权重 (默认: 0.4)')
    parser.add_argument('--simhash_bits', type=int, default=64,
                       help='SimHash位数 (默认: 64)')
    
    args = parser.parse_args()
    
    # 创建配置
    config = RerankerConfig(
        simhash_bits=args.simhash_bits,
        alpha=args.alpha,
        beta=args.beta,
        use_3gram=True,
        normalize_scores=True
    )
    
    # 创建批量重排序器
    reranker = MultiDatasetReranker(config)
    
    # 处理所有数据集
    summary = reranker.process_all_datasets()


if __name__ == "__main__":
    main() 