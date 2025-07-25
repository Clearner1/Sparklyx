from sparkly.utils import get_logger, norm_auc
from scipy import stats
from abc import ABC, abstractmethod

import numpy as np
from copy import deepcopy

log = get_logger(__name__)


def score_query_results(query_results):
    return [score_query_result(t.scores) for t in query_results]

def score_query_result(scores, drop_first=False):
    scores = np.array(scores)
    if len(scores.shape) == 0:
        return 1.0

    if drop_first:
        scores = scores[1:]

    if len(scores) < 2:
        return 1.0
    else:
        return norm_auc(scores / scores[0])

def score_query_result_sum(scores):
    scores = np.array(scores)
    if len(scores.shape) == 0 or len(scores) < 2:
        return 0.0
    else:
        return scores.sum() / (norm_auc(scores / scores[0]) ** 2)

def _update_spec(base, field, path):
        s = deepcopy(base)
        paths = s.get(field, []) 
        # don't add redundant paths
        if path not in paths:
            paths.append(path)
        s[field] = paths
        return s

def compute_wilcoxon_score(x,y):
    z = x  -y 
    # score is 0 if all elements are the same
    # pval is 1
    if (z == 0).all():
        return (0, 1)
    else:
        return stats.wilcoxon(z)

class QueryScorer(ABC):
    
    # lower score means better query 
    @abstractmethod
    def score_query_results(self, query_results, query_spec) -> list:
        pass
    @abstractmethod
    def score_query_result(self, query_result, query_spec) -> float:
        pass


class AUCQueryScorer(QueryScorer):

    def __init__(self):
        pass

    def score_query_results(self, query_results, query_spec, drop_first) -> list:
        return [self.score_query_result(r, query_spec, drop_first) for r in query_results]

    def score_query_result(self, query_result, query_spec, drop_first) -> float:
        return score_query_result(query_result.scores, drop_first)



class RankQueryScorer(QueryScorer):

    def __init__(self, threshold, k):
        self._threshold = threshold
        self._k = k

    def score_query_results(self, query_results, query_spec) -> list:
        return [self.score_query_result(r, query_spec) for r in query_results]

    def score_query_result(self, query_result, query_spec) -> float:
        scores = query_result.scores
        if len(scores.shape) == 0 or len(scores) < 2:
            return self._k
        scores = scores / scores[0]
        return np.searchsorted(scores, self._threshold, side='right')
        

        return score_query_result(query_result.scores)
    
class WeightedAUCQueryScorer(AUCQueryScorer):
    """考虑字段权重的评分器"""
    
    def __init__(self):
        super().__init__()

    def score_query_results(self, query_results, query_spec, drop_first=False) -> list:
        """评估带权重的查询结果
        
        Parameters
        ----------
        query_results : list
            查询结果列表
        query_spec : QuerySpec
            查询规格
        drop_first : bool
            是否丢弃第一个结果
            
        Returns
        -------
        list
            评分结果列表
        """
        # 先用基础的AUC评分
        base_scores = super().score_query_results(query_results, query_spec, drop_first)
        
        if query_spec is None:
            return base_scores
            
        # 获取涉及字段的权重
        field_weights = {}
        for field, _ in query_spec.items():
            base_field = field.split('.')[0]
            if base_field not in field_weights:
                field_weights[base_field] = query_spec._config.field_weights.get(base_field, 1.0)
        
        # 如果没有权重信息，返回基础分数
        if not field_weights:
            return base_scores
            
        # 计算加权平均的字段权重
        avg_weight = sum(field_weights.values()) / len(field_weights)
        
        # 调整评分结果
        weighted_scores = [score * avg_weight for score in base_scores]
        
        return weighted_scores

    def score_query_result(self, query_result, query_spec, drop_first=False) -> float:
        """评估单个带权重的查询结果"""
        # 获取基础评分
        base_score = super().score_query_result(query_result, query_spec, drop_first)
        
        if query_spec is None:
            return base_score
        
        # 获取涉及字段的权重
        field_weights = {}
        for field, _ in query_spec.items():
            base_field = field.split('.')[0]
            if base_field not in field_weights:
                field_weights[base_field] = query_spec._config.field_weights.get(base_field, 1.0)
                
        # 如果没有权重信息，返回基础分数
        if not field_weights:
            return base_score
            
        # 计算加权平均的字段权重
        avg_weight = sum(field_weights.values()) / len(field_weights)
        
        # 返回调整后的分数
        return base_score * avg_weight