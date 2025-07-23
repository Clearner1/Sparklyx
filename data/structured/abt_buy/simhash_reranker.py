#!/usr/bin/env python3
"""
SimHash重排序器 - 基于SimHash特征的搜索结果重排序

主要功能:
1. 从优化结果中加载字段权重
2. 基于加权字段提取SimHash特征
3. 计算查询与候选文档的SimHash相似度
4. 融合BM25分数和SimHash相似度进行重排序
5. 评估重排序效果

作者: Zane
日期: 2025
"""

import json
import pandas as pd
import numpy as np
import hashlib
from typing import List, Dict, Tuple, Optional
from collections import Counter
import re
from dataclasses import dataclass
import time


@dataclass
class RerankerConfig:
    """重排序器配置"""
    simhash_bits: int = 64  # SimHash位数
    alpha: float = 0.7      # BM25权重
    beta: float = 0.3       # SimHash权重
    use_3gram: bool = True  # 是否使用3-gram特征
    normalize_scores: bool = True  # 是否标准化分数


class TextAnalyzer:
    """文本分析器 - 复现Sparkly的分析器功能"""
    
    def __init__(self, use_3gram: bool = True):
        self.use_3gram = use_3gram
        
    def standard_tokenize(self, text: str) -> List[str]:
        """标准分词器 - 基于空格和标点符号"""
        if not text:
            return []
        # 转小写，移除特殊字符，按空格分词
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return [token for token in tokens if len(token) > 0]
    
    def gram3_tokenize(self, text: str) -> List[str]:
        """3-gram分词器"""
        if not text:
            return []
        # 移除非字母数字字符，转小写
        text = re.sub(r'[^a-zA-Z0-9]', '', text.lower())
        if len(text) < 3:
            return [text] if text else []
        
        grams = []
        for i in range(len(text) - 2):
            gram = text[i:i+3]
            if gram.isalnum():  # 只保留字母数字3-gram
                grams.append(gram)
        return grams
    
    def analyze(self, text: str, analyzer_type: str = 'standard') -> List[str]:
        """分析文本，返回token列表"""
        if not text:
            return []
            
        if analyzer_type == 'standard':
            return self.standard_tokenize(text)
        elif analyzer_type == '3gram':
            return self.gram3_tokenize(text)
        else:
            raise ValueError(f"Unsupported analyzer type: {analyzer_type}")


class SimHashExtractor:
    """SimHash特征提取器（通用版，自动适配字段）"""
    
    def __init__(self, config: RerankerConfig, field_weights: Dict[str, float], concat_fields: Dict[str, list], field_to_analyzers: Dict[str, list], default_analyzer: str = 'standard'):
        self.config = config
        self.field_weights = field_weights
        self.concat_fields = concat_fields or {}
        self.field_to_analyzers = field_to_analyzers or {}
        self.default_analyzer = default_analyzer
        self.analyzer = TextAnalyzer(config.use_3gram)

    def extract_weighted_tokens(self, record: Dict) -> List[Tuple[str, float]]:
        """提取加权token，自动适配所有字段和拼接字段"""
        weighted_tokens = []
        # 1. 普通字段
        for field, weight in self.field_weights.items():
            if field in self.concat_fields.keys():
                continue  # 拼接字段后面处理
            text = str(record.get(field, ""))
            if not text or text == "nan":
                continue
            analyzers = self.field_to_analyzers.get(field, [self.default_analyzer])
            for analyzer_type in analyzers:
                # 只支持standard/3gram
                if '3gram' in analyzer_type:
                    tokens = self.analyzer.analyze(text, '3gram')
                    prefix = f"{field}.3gram"
                else:
                    tokens = self.analyzer.analyze(text, 'standard')
                    prefix = f"{field}.standard"
                for token in tokens:
                    weighted_tokens.append((f"{prefix}.{token}", weight))
        # 2. 拼接字段
        for concat_field, sub_fields in self.concat_fields.items():
            weight = self.field_weights.get(concat_field, 0.0)
            if weight <= 0:
                continue
            concat_text = " ".join([str(record.get(f, "")) for f in sub_fields if record.get(f, "") and str(record.get(f, "")) != "nan"])
            if not concat_text.strip():
                continue
            analyzers = self.field_to_analyzers.get(concat_field, [self.default_analyzer])
            for analyzer_type in analyzers:
                if '3gram' in analyzer_type:
                    tokens = self.analyzer.analyze(concat_text, '3gram')
                    prefix = f"{concat_field}.3gram"
                else:
                    tokens = self.analyzer.analyze(concat_text, 'standard')
                    prefix = f"{concat_field}.standard"
                for token in tokens:
                    weighted_tokens.append((f"{prefix}.{token}", weight))
        return weighted_tokens
    
    def compute_simhash(self, record: Dict) -> int:
        """计算记录的SimHash值"""
        weighted_tokens = self.extract_weighted_tokens(record)
        
        if not weighted_tokens:
            return 0
        
        # 初始化特征向量
        feature_vector = [0.0] * self.config.simhash_bits
        
        # 对每个加权token进行处理
        for token, weight in weighted_tokens:
            # 计算token的哈希值
            hash_value = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
            
            # 对于每一位，根据哈希值的对应位来更新特征向量
            for i in range(self.config.simhash_bits):
                bit = (hash_value >> i) & 1
                if bit == 1:
                    feature_vector[i] += weight
                else:
                    feature_vector[i] -= weight
        
        # 生成最终的SimHash
        simhash = 0
        for i in range(self.config.simhash_bits):
            if feature_vector[i] > 0:
                simhash |= (1 << i)
        
        return simhash


class SimHashReranker:
    """SimHash重排序器主类（自动适配所有数据集）"""
    
    def __init__(self, config: RerankerConfig = None):
        self.config = config or RerankerConfig()
        self.field_weights = {}
        self.concat_fields = {}
        self.field_to_analyzers = {}
        self.default_analyzer = 'standard'
        self.extractor = None
        
    def load_optimization_result(self, opt_result_path: str):
        """加载优化结果，自动获取字段权重、拼接字段、分析器类型"""
        with open(opt_result_path, 'r', encoding='utf-8') as f:
            opt_result = json.load(f)
        self.field_weights = opt_result['field_weights']
        index_cfg = opt_result.get('index_config_summary', {})
        self.concat_fields = index_cfg.get('concat_fields', {})
        self.field_to_analyzers = index_cfg.get('field_to_analyzers', {})
        self.default_analyzer = index_cfg.get('default_analyzer', 'standard')
        print(f"加载字段权重: {self.field_weights}")
        print(f"拼接字段: {self.concat_fields}")
        print(f"字段分析器: {self.field_to_analyzers}")
        # 初始化特征提取器
        self.extractor = SimHashExtractor(self.config, self.field_weights, self.concat_fields, self.field_to_analyzers, self.default_analyzer)
        
    def hamming_distance(self, hash1: int, hash2: int) -> int:
        """计算汉明距离"""
        return bin(hash1 ^ hash2).count('1')
    
    def simhash_similarity(self, hash1: int, hash2: int) -> float:
        """计算SimHash相似度 (0-1之间，1表示完全相同)"""
        distance = self.hamming_distance(hash1, hash2)
        similarity = 1.0 - (distance / self.config.simhash_bits)
        return max(0.0, similarity)
    
    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """标准化分数到0-1范围"""
        if len(scores) == 0:
            return scores
        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)
    
    def rerank_candidates(self, 
                         query_record: Dict, 
                         candidate_records: List[Dict], 
                         original_scores: List[float]) -> Tuple[List[int], List[float], Dict]:
        """重排序候选结果
        
        Args:
            query_record: 查询记录
            candidate_records: 候选记录列表
            original_scores: 原始BM25分数
            
        Returns:
            (重排序后的索引, 融合分数, 调试信息)
        """
        if not self.extractor:
            raise ValueError("请先调用load_optimization_result加载配置")
        
        start_time = time.time()
        
        # 1. 计算查询记录的SimHash
        query_simhash = self.extractor.compute_simhash(query_record)
        
        # 2. 计算所有候选记录的SimHash和相似度
        simhash_similarities = []
        candidate_simhashes = []
        
        for candidate in candidate_records:
            candidate_simhash = self.extractor.compute_simhash(candidate)
            candidate_simhashes.append(candidate_simhash)
            similarity = self.simhash_similarity(query_simhash, candidate_simhash)
            simhash_similarities.append(similarity)
        
        # 3. 标准化分数
        original_scores = np.array(original_scores)
        simhash_similarities = np.array(simhash_similarities)
        
        if self.config.normalize_scores:
            norm_bm25 = self.normalize_scores(original_scores)
            norm_simhash = simhash_similarities  # SimHash相似度已经在0-1范围内
        else:
            norm_bm25 = original_scores
            norm_simhash = simhash_similarities
        
        # 4. 分数融合
        fused_scores = (self.config.alpha * norm_bm25 + 
                       self.config.beta * norm_simhash)
        
        # 5. 重排序
        ranking_indices = np.argsort(-fused_scores)  # 降序排列
        reranked_scores = fused_scores[ranking_indices]
        
        # 6. 准备调试信息
        processing_time = time.time() - start_time
        debug_info = {
            'query_simhash': query_simhash,
            'candidate_simhashes': candidate_simhashes,
            'original_scores': original_scores.tolist(),
            'simhash_similarities': simhash_similarities.tolist(),
            'fused_scores': fused_scores.tolist(),
            'processing_time': processing_time,
            'config': {
                'alpha': self.config.alpha,
                'beta': self.config.beta,
                'simhash_bits': self.config.simhash_bits
            }
        }
        
        return ranking_indices.tolist(), reranked_scores.tolist(), debug_info


class RerankerEvaluator:
    """重排序效果评估器"""
    
    def __init__(self):
        pass
    
    def load_ground_truth(self, gold_path: str) -> Dict[int, int]:
        """加载真实标签"""
        with open(gold_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        gold_mapping = {}
        for line in lines:
            if line.strip():
                data = json.loads(line.strip())
                # id2 (table_b) -> id1 (table_a)
                gold_mapping[data['id2']] = data['id1']
        
        return gold_mapping
    
    def calculate_metrics(self, 
                         query_id: int,
                         original_ranking: List[int],
                         reranked_ranking: List[int],
                         gold_mapping: Dict[int, int],
                         k_values: List[int] = [1, 5, 10, 20]) -> Dict:
        """计算评估指标"""
        
        if query_id not in gold_mapping:
            return {'has_ground_truth': False}
        
        true_match_id = gold_mapping[query_id]
        
        metrics = {'has_ground_truth': True, 'true_match_id': true_match_id}
        
        # 计算不同k值下的Recall@k和排名变化
        for k in k_values:
            # 原始排序
            orig_topk = original_ranking[:k]
            orig_recall = 1.0 if true_match_id in orig_topk else 0.0
            orig_rank = orig_topk.index(true_match_id) + 1 if true_match_id in orig_topk else -1
            
            # 重排序后
            rerank_topk = reranked_ranking[:k]
            rerank_recall = 1.0 if true_match_id in rerank_topk else 0.0
            rerank_rank = rerank_topk.index(true_match_id) + 1 if true_match_id in rerank_topk else -1
            
            metrics[f'original_recall@{k}'] = orig_recall
            metrics[f'reranked_recall@{k}'] = rerank_recall
            metrics[f'original_rank@{k}'] = orig_rank
            metrics[f'reranked_rank@{k}'] = rerank_rank
            metrics[f'rank_improvement@{k}'] = orig_rank - rerank_rank if orig_rank > 0 and rerank_rank > 0 else 0
        
        return metrics


if __name__ == "__main__":
    # 基本使用示例
    print("SimHash重排序器已准备就绪!")
    print("使用方法:")
    print("1. 创建重排序器: reranker = SimHashReranker()")
    print("2. 加载配置: reranker.load_optimization_result('optimization_result.json')")
    print("3. 重排序: ranking, scores, debug = reranker.rerank_candidates(query, candidates, scores)") 