# Dirty数据集批量重排序说明

## 📊 数据集概览

dirty文件夹包含4个数据集：

1. **amazon-google** - Amazon vs Google产品匹配
2. **dblp-acm** - DBLP vs ACM学术论文匹配  
3. **dblp-googlescholar** - DBLP vs Google Scholar论文匹配
4. **walmart-amazon** - Walmart vs Amazon产品匹配

所有数据集都包含完整的文件：
- `sparkly_results_k50.parquet` - Sparkly搜索结果
- `table_a.parquet` - 被索引数据
- `table_b.parquet` - 查询数据
- `optimization_result.json` - 字段权重配置
- `gold.parquet` - 真实标签

## 🚀 执行重排序

### 切换到目录并运行：

```bash
# 进入dirty数据集目录
cd data/dirty/

# 使用默认参数运行（推荐）
python batch_rerank_all_datasets.py

# 或者自定义参数
python batch_rerank_all_datasets.py --alpha 0.7 --beta 0.3

# 查看所有参数选项
python batch_rerank_all_datasets.py --help
```

### 参数说明：

- `--alpha` (默认: 0.6) - BM25权重，推荐范围0.5-0.8
- `--beta` (默认: 0.4) - SimHash权重，推荐范围0.2-0.5  
- `--simhash_bits` (默认: 64) - SimHash位数，32或64

## 📄 输出文件

执行完成后会生成：

### 单个数据集结果：
- `amazon-google_reranked_results_k50.parquet` 
- `amazon-google_evaluation_report.json`
- `dblp-acm_reranked_results_k50.parquet`
- `dblp-acm_evaluation_report.json` 
- `dblp-googlescholar_reranked_results_k50.parquet`
- `dblp-googlescholar_evaluation_report.json`
- `walmart-amazon_reranked_results_k50.parquet`
- `walmart-amazon_evaluation_report.json`

### 汇总报告：
- `all_datasets_rerank_summary.json` - 详细分析报告
- `all_datasets_rerank_summary.csv` - 表格格式汇总

## 📈 论文用途

这些结果非常适合论文写作：

1. **对比实验表格** - CSV文件可直接用于论文表格
2. **多数据集验证** - 4个不同领域的数据集验证方法的泛化性
3. **统计显著性** - 包含详细的改善统计和查询级别分析
4. **计算开销分析** - 重排序时间统计

## 🎯 预期效果

Dirty数据集通常比Structured数据集更具挑战性，因为：
- 数据质量较低
- 噪声更多
- 匹配难度更大

重排序在这些数据集上可能会有更明显的效果改善！

## ⚠️ 注意事项

1. **处理时间** - 根据数据集大小，可能需要数分钟到数小时
2. **内存使用** - 确保有足够内存加载大型数据集
3. **依赖模块** - 确保能访问`../structured/abt_buy/simhash_reranker.py`

## 🔍 快速检查

运行前可以快速检查数据：

```bash
python -c "
import pandas as pd
import os
for dataset in ['amazon-google', 'dblp-acm', 'dblp-googlescholar', 'walmart-amazon']:
    if os.path.exists(dataset):
        df = pd.read_parquet(f'{dataset}/sparkly_results_k50.parquet')
        print(f'{dataset}: {len(df)} 个查询')
"
``` 