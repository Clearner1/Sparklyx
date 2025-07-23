# 命令行使用指南

本文档介绍了如何使用 `test_rerank_ag.py` 脚本来对数据集进行重排序评估。

## 基本用法

```bash
python test_rerank_ag.py [--alpha 0.6] [--beta 0.4] [--sample_size 100]
```

## 命令行参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--alpha` | float | 0.6 | BM25 权重 |
| `--beta` | float | 0.4 | SimHash 权重 |
| `--simhash_bits` | int | 64 | SimHash 位数 |
| `--sample_only` | flag | - | 只处理样本查询，用于快速测试 |
| `--sample_size` | int | 100 | 样本大小 |

## 使用示例

### 1. 处理全部查询（默认）

```bash
python test_rerank_ag.py
```

### 2. 处理样本查询

```bash
python test_rerank_ag.py --sample_only
```

### 3. 使用自定义参数处理样本查询

```bash
python test_rerank_ag.py --sample_only --alpha 0.7 --beta 0.3 --sample_size 50
```

### 4. 使用不同权重处理全部查询

```bash
python test_rerank_ag.py --alpha 0.5 --beta 0.5
```

## 输出文件

| 文件名 | 描述 |
|--------|------|
| `{dataset}_reranked_results_k50.parquet` | 重排序结果 |
| `{dataset}_evaluation_report.json` | 详细评估报告 |
| `{dataset}_sample_cases.json` | 典型案例分析（仅在样本模式下生成） |
| `{dataset}_sample_reranked_results.parquet` | 样本重排序结果（仅在样本模式下生成） |
| `{dataset}_sample_evaluation_report.json` | 样本评估报告（仅在样本模式下生成） |

注：`{dataset}` 会被自动替换为当前目录的名称。