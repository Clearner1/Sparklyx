from itertools import islice, combinations
from scipy import stats
from joblib import delayed
from pyspark import SparkContext
import pyspark.sql.functions as F
import pyspark
import numpy as np
from sparkly.index import Index
from sparkly.index_config import IndexConfig
from sparkly.query_generator import QuerySpec
import pandas as pd
import math
from copy import deepcopy
from sparkly.index.lucene_index import LuceneIndex
from sparkly.utils import  get_logger, invoke_task, type_check, type_check_call
from sparkly.index_optimizer.query_scorer import AUCQueryScorer, QueryScorer, WeightedAUCQueryScorer
from sparkly.search import search
from typing import Annotated
from pydantic import Field

pd.set_option('display.width', 150)

log = get_logger(__name__)


def _compute_wilcoxon_score(x,y):
    z = x  - y 
    # score is 0 if all elements are the same
    # pval is 1
    if (z == 0).all():
        return (0, 1)
    else:
        return stats.wilcoxon(z)
        
def adaptive_continuity_correction_wilcoxon(x, y):
    """
    改进的Wilcoxon符号秩检验
    主要改进：根据样本特征自适应调整连续性校正因子
    - 小样本(≤10): 使用0.3校正，提高敏感性
    - 中样本(11-25): 使用0.4校正，平衡精度与稳健性
    - 大样本(>25): 根据重复值情况使用0.4-0.5校正
    
    Parameters:
    -----------
    x, y: array-like
        两个配置的AUC分数向量
        
    Returns:
    --------
    tuple: (统计量, p值)
    """
    from scipy.stats import wilcoxon
    import numpy as np
    
    differences = x - y
    n = len(differences[differences != 0])
    
    # 核心改进：根据样本量和数据分布调整连续性校正因子
    if n <= 10:
        correction = 0.3  # 小样本用更小的校正
    elif n <= 25: 
        correction = 0.4
    else:
        correction = 0.5  # 大样本用标准校正
    
    # 根据数据的离散程度进一步调整
    if len(np.unique(differences)) < n * 0.7:  # 有很多重复值
        correction *= 0.8
    
    # 手动计算Wilcoxon统计量（为了应用自定义校正）
    abs_diffs = np.abs(differences[differences != 0])
    ranks = rank_data_with_ties(abs_diffs)
    
    positive_ranks = ranks[differences[differences != 0] > 0]
    W_plus = np.sum(positive_ranks)
    
    # 应用自适应连续性校正
    E_W = n * (n + 1) / 4
    Var_W = n * (n + 1) * (2 * n + 1) / 24
    
    z = (W_plus - E_W - correction) / np.sqrt(Var_W)
    
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z)))
    
    return W_plus, p_value

class IndexOptimizer():
    """
    a class for optimizing the search columns and analyzers for indexes
    """
    @type_check_call
    def __init__(self, 
            is_dedupe: bool,
            scorer: QueryScorer | None=None,
            conf: Annotated[float, Field(ge=0, lt=1.0)]=.99, 
            init_top_k: int=10,
            max_combination_size: int=3,
            opt_query_limit: int=250,
            sample_size: int=10000,
            use_early_pruning: bool=True
        ):
        """
        Parameters
        ----------
        is_dedupe : bool
            Should be true if the search table == the indexed table

        scorer : QueryScorer, optional
            the class that will be used to score the queries during optimization.
            If not provided defaults to AUCQueryScorer

        conf : float, default=.99
            the confidence score cut off during optimization
            查询评分器，如果为None则默认使用WeightedAUCQueryScorer
        """

        if conf < 0 or conf >= 1:
            raise ValueError(f'conf must be in the interval [0, 1), (got {conf})')

        self._scorer = scorer if scorer is not None else AUCQueryScorer()
        self._index = None
        self._confidence = conf
        self._is_dedupe = is_dedupe
        self._search_chunk_size = 50
        self._opt_query_limit = opt_query_limit
        self._init_top_k = init_top_k
        self._max_combination_size = max_combination_size
        self._sample_size = sample_size
        self._use_early_pruning = use_early_pruning 

        self._auc_sim = {
            'type' : 'BM25',
            'b' : .75,
            'k1' : 1.2,
        }

    
    @property
    def index(self):
        return self._index

    @index.setter
    @type_check_call
    def index(self, i: Index):
        self._index = i
        self._index.to_spark()

    def _generate_cand_query_specs(self, bases, cands):
        if isinstance(bases, QuerySpec):
            return [bases.union(c) for c in cands]
        else:
            return list({b.union(c) for b in bases for c in cands})

    @staticmethod
    def _count_empty_queries(spec, nulls):
        cols = [c for c in spec if c in nulls.columns]
        # FIXME this is just assuming that concat columns don't contain nulls
        if len(cols) != len(spec):
            return 0

        return nulls['count'].loc[nulls[cols].all(axis=1)].sum()


    def _get_min_tasks(self):
        return SparkContext.getOrCreate().defaultParallelism * 4

    def _execute_and_score(self, tasks : pd.Series):
        sc = SparkContext.getOrCreate()
        res = sc.parallelize(tasks.values, len(tasks))\
                .map(lambda x : self._scorer.score_query_results(invoke_task(x), None, self._is_dedupe))\
                .collect()
        
        return pd.Series(res, index=tasks.index)

    
    def _get_nulls(self, search_df):
        cols = [c for c in search_df.columns if c != '_id']

        return search_df.select([F.col(c).isNull().alias(c) for c in cols])\
                        .groupby(*cols)\
                        .count()\
                        .toPandas()

    
    def _iter_slices_df(self, df, slice_size):
        for start in range(0, len(df), slice_size):
            end = start + self._search_chunk_size
            recs = df.iloc[start:end]
            yield recs

        if start < len(df):
            recs = df.iloc[start:]
            yield recs

    def _gen_search_tasks(self, search_df, query_spec=None, id_col='_id'):
        
        for search_recs in self._iter_slices_df(search_df, self._search_chunk_size):
            search_recs = search_recs.where(pd.notnull(search_recs), None)\
                                    .astype(object)\
                                    .set_index(id_col)\
                                    .to_dict('records')

            yield delayed(search)(self._index, query_spec, self._opt_query_limit, search_recs)


    def _get_topk_specs(self, cands, search_df, nulls, k=1, null_top_k=20, early_term_thres=math.inf):
        """
        查找产生最佳搜索结果的前 k 个查询规格（specs）。

        Args:
            cands: 候选查询规格的 DataFrame。必须包含一个 'spec' 列。
            search_df: 包含搜索数据的 DataFrame。
            nulls: 不应返回任何结果的查询集合。用于过滤候选规格。
            k: 要返回的最佳规格数量。
            null_top_k: 如果某些规格产生空查询，则要考虑的候选规格的最大数量。
            early_term_thres:  提前终止的阈值。此特定实现中未使用。

        Returns:
            包含前 k 个查询规格的 DataFrame。
        """
        # 如果 k 大于等于候选规格的数量，则返回所有候选规格。
        if k >= len(cands):
            cands['mean'] = 0.0
            cands['std'] = 0.0
            return cands
        
        if null_top_k > 0:
            # 计算每个规格产生的空查询数量。
            cands['num_empty_queries'] = cands['spec'].apply(IndexOptimizer._count_empty_queries, nulls=nulls)
            cands.sort_values('num_empty_queries', inplace=True)
            if cands['num_empty_queries'].eq(0).sum() < null_top_k:
                # if there are less than null top-k specs that produce no empty queries, 
                # take the top 20 than produce as few empty queries as possible
                cands = cands.head(null_top_k)
                log.info(f' < {null_top_k} candidates produce 0 empty queries, taking top-k')
                log.info(f'top-{null_top_k} cands :\n{cands}') 
            else:
                # 否则，取所有不产生空查询的规格
                cands = cands.loc[cands['num_empty_queries'].eq(0)]

            if len(cands) == 0:
                raise RuntimeError('no candidates left after removing those that contain null')
        # 为每个剩余的候选规格生成搜索任务。
        task_queues = cands['spec'].apply(lambda x : self._gen_search_tasks(query_spec=x, search_df=search_df))
        # 初始化一个空的 DataFrame 来存储每个规格的得分。
        scores = pd.DataFrame([], columns=task_queues.index)

        while True:
            # 确定每次迭代要运行的最小任务数。
            min_tasks = self._get_min_tasks() if self._use_early_pruning else len(search_df)
            # 计算本次迭代中每个队列要运行的任务数。
            n_tasks_per_queue = max(5, min_tasks // len(task_queues))
            log.debug(f'n_queues : {len(task_queues)}, min_tasks : {min_tasks}, n_tasks_per_queue : {n_tasks_per_queue}')
            task_grp = task_queues.apply(lambda x : list(islice(x, n_tasks_per_queue)))
            # any of the iterators are exhasusted, terminate
            if task_grp.apply(len).eq(0).any() == True:
                log.debug('tasks exhausted, returning min')
                break
            # get the next slice of tasks to run seach for 
            # run search for tasks
            # list of lists with query results (id2, id1_list, scores)
            task_grp = task_grp.explode()

            # 执行搜索任务并对结果进行评分。
            res = self._execute_and_score(task_grp)\
                 .explode()\
                .groupby(level=0)\
                .apply(lambda x : x.values)

            # 将结果转换为 DataFrame。
            res = pd.DataFrame(res.to_dict(), dtype=np.float64)
            # 将新结果添加到当前得分中。
            scores = pd.concat([scores, res], ignore_index=True)
            # 计算平均得分和标准差。
            score_stats = scores.apply(lambda x : pd.Series({
                                                    'mean' : np.mean(x),
                                                    'std' : x.std()
                                                }))\
                                .transpose()\

            score_stats['spec'] = cands['spec']
            score_stats = score_stats.sort_values('mean')
            log.debug(f'score_stats\n{score_stats.to_string()}')

            # get minimum rows
            topk = score_stats.head(k)
            # running list of all the specs that are less 
            # than EVERYTHING in topk 
            # these can be safely dropped at the end 
            drop_specs = score_stats.index.values[k:]

            # 对每个 topk 规格执行 Wilcoxon 秩和检验。
            for i in range(len(topk)):
                min_row = topk.iloc[i]
                # compute the wilcoxon scores for the rows against the row 
                # with the minimum mean
                wc_scores = scores.drop(columns=topk.iloc[:i+1].index)\
                                    .apply(lambda x : adaptive_continuity_correction_wilcoxon(scores[min_row.name], x))\
                                    .transpose()
            
                wc_scores.columns = ['statistic', 'pval']
                log.debug(f'wilcoxon scores\n{wc_scores}')
                # all the query specs that should be excluded 
                # because they have a score that is greater
                exclude_cols = wc_scores['pval'].lt(1 - self._confidence)
                drop_specs = np.intersect1d(drop_specs, exclude_cols[exclude_cols.values].index.values)
            
            if len(drop_specs) == len(score_stats) - k:
                # everything was statistically significant
                break
            else:
                # remove specs that compared less to everything else
                scores.drop(columns=drop_specs, inplace=True)
                task_queues.drop(index=drop_specs, inplace=True)




        return topk
    
    def _union_specs(self, specs):
        s = specs[0]
        for o in specs[1:]:
            s = s.union(o)
        return s

    def _gen_combs(self, specs, max_k=3):
        if max_k is None:
            max_k = len(specs)

        out = list(specs)
        for k in range(2, max_k+1):
            out.extend(map(self._union_specs, combinations(specs, k)))

        return out



    def _has_overlapping_fields(self, spec):
        # check that the spec doesn't have any fields that overlap
        concat_fields = self.index.config.concat_fields
        for f in spec:
            if f in concat_fields:
                for v in concat_fields[f]:
                    if v in spec:
                        return True
        # check for sets of paths which have overlapping 
        for paths in spec.values():
            paths = {s.split('.')[0] for s in paths}
            for f in paths:
                if f in concat_fields:
                    for v in concat_fields[f]:
                        if v in paths:
                            return True

        return False

    
    
    def _sample_df(self, search_df, nulls):
        if self._sample_size is not None:
            search_df =  search_df.limit(self._sample_size)
        return search_df.toPandas()
    
    def _count_average_tokens(self, df):
        cols = [F.size(F.split(F.col(c).cast("string"), "\\s+")).alias(c) for c in df.columns if c != '_id']
        df = df.select(cols).toPandas()
        df = df.where(df >= 0)
        return df.mean()

    @type_check_call
    def make_index_config(self, df: pyspark.sql.DataFrame, id_col='_id') -> IndexConfig:
        """
        create the starting index config which can then be used to for optimization
        throws out any columns where the average number of 
        whitespace delimited tokens are >= 50

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            the dataframe that we want to generate a config for
        id_col : str
            the unique id column for the records in the dataframe
        """

        analyzers = [
                'standard',
                '3gram',
        ]
        index_config = IndexConfig()
        
        counts = self._count_average_tokens(df)
        log.debug(f'token counts :\n{counts}')
        # drop long columns 去掉列(内容平均超过50的)
        columns = counts.index[counts <= 50].tolist()
        if len(columns) == 0:
            raise RuntimeError('all columns dropped due to length')

        log.debug(f'columns for config : {columns}')


        for c in columns:
            if c == id_col:
                continue
            index_config.add_field(c, analyzers)

        # 当一个数据库有多个列，则自动连接其他列
        if len(columns) > 1:       
            name = 'concat_' + '_'.join(sorted(columns))
            index_config.add_concat_field(name, columns, analyzers)

        # 设置相似度配置
        index_config.sim = deepcopy(self._auc_sim)

        return index_config

    
    @type_check_call
    def optimize(self, index : Index, search_df: pyspark.sql.DataFrame) -> QuerySpec:
        """

        Parameters
        ----------

        index : Index
            the index that will have an optimzed query spec created for it
            
        search_df : pyspark.sql.DataFrame:
            the records that will be used to choose the query spec

        Returns
        -------

        QuerySpec
            a query spec optimized for searching for `search_df` using `index`

        """
        self.index = index
        # 评估过程
        nulls = self._get_nulls(search_df)
        #  1. 数据采样
        search_df = self._sample_df(search_df, nulls)
        # 2. 获取所有可能的查询配置
        full_query_spec = self.index.get_full_query_spec(cross_fields=True)

        # generate single column query specs
        # 3. 单字段评估
        single_specs =  pd.DataFrame({
            'spec' : [QuerySpec({k : [v]})
                       for k,p in full_query_spec.items()
                         for v in p]
        })

        log.debug(f'starting candidate query specs\n{single_specs}')
        # get the top-k starting single query specs 经验值10
        # 评估单字段性能
        start_k = self._init_top_k
        # 4. 选择前k个最优的配置
        single_specs = self._get_topk_specs(single_specs, search_df, null_top_k=-1, k=start_k, nulls=nulls)

        single_specs = single_specs.loc[single_specs['mean'] < 1.0]

        log.debug(f'top-{start_k} starting candidate query specs\n{single_specs.to_string()}')

        # 计算字段权重
        field_weights = self._calculate_field_weights(single_specs)
        log.debug(f"Field weights before setting: {field_weights}")
        # 临时解冻配置，更新权重，然后重新冻结
        self.index.config.unfreeze()
        self.index.config.field_weights = field_weights
        log.debug(f"Config field weights after setting: {self.index.config.field_weights}")
        self.index.config.freeze()

        import json
        log.info("Saving field weights to field_weights.json")
        with open('field_weights.json', 'w') as f:
            json.dump(field_weights, f, indent=4)

        print("\n=== Field Weights ===")
        for field, weight in self.index.config.field_weights.items():
            print(f"{field}: {weight:.4f}")
        print("===================\n")
        # generate combinations of query specs       
        # 5. 生成字段组合
        extensions = self._gen_combs(
            single_specs['spec'].values, 
            max_k=self._max_combination_size
            )
        
        extensions = [s for s in extensions 
                    if not self._has_overlapping_fields(s)]
        
        # 优化迭代
        max_depth = 1
        # optimization state
        min_score = math.inf
        best_query_specs = [QuerySpec()]
        # maximum number of opt iterations
        # defaults to no limit
        #max_depth = max_depth if max_depth is not None else len(cands)
        # main optimization loop # 6. 迭代优化组合
        for i in range(max_depth):
            # add a new path to the current best
            cands = pd.DataFrame({'spec' : self._generate_cand_query_specs(best_query_specs, extensions)})
            cands = cands.loc[~cands['spec'].apply(self._has_overlapping_fields)]
            # add the current best for comparision
            # skip any columns that contain null on the first iteration
            # this is because we want to ensure that every record has a value to 
            # index/block on

            # take the top spec if this is the last iteration, else the top 10
            k = 1 if i == max_depth - 1 else 10
            top_specs_stats = self._get_topk_specs(cands, search_df, k=k, nulls=nulls)
            top_spec = top_specs_stats.iloc[0]

            log.debug(f'best spec = {top_spec}')
            cand_score = top_spec['mean']
            
            if  cand_score >= min_score:
                # no improvement this iteration, terminate
                log.debug('cand_score >= min_score, end search')
                break
            else:
                # prevent warning being raised on the first iteration
                if not math.isinf(min_score):
                    log.debug(f'% improvement : {(min_score - cand_score) / min_score}')
                # update current min
                min_score = cand_score
                best_query_specs = top_specs_stats['spec'].tolist()
                log.debug('cand_score < min_score, replacing')


        return best_query_specs[0]

    # 计算字段权重的辅助方法
    def _calculate_field_weights(self, single_specs):
        """基于meanAUC计算字段权重
        
        Parameters
        ----------
        single_specs : pd.DataFrame
            包含每个字段-分析器组合的meanAUC评分
            
        Returns
        -------
        Dict[str, float]
            字段到权重的映射
        """
        # 获取基础字段名（去掉分析器后缀）
        single_specs['base_field'] = single_specs['spec'].apply(
            lambda x: list(x.keys())[0].split('.')[0]
        )
        
        # 对每个基础字段，选择最佳的meanAUC分数
        field_scores = single_specs.groupby('base_field')['mean'].min()
        log.debug(f"Field scores:\n{field_scores}")  # 添加日志

        # 转换分数为权重（分数越小越好）
        max_score = field_scores.max()
        min_score = field_scores.min()

        # 使用归一化的线性转换
        normalized_scores = (max_score - field_scores) / (max_score - min_score)
        log.debug(f"Normalized scores:\n{normalized_scores}")

        # 使用 softmax 来增加差异
        exp_scores = np.exp(normalized_scores * 5)  # 乘以5来增加差异
        weights = exp_scores / exp_scores.sum()

        result = weights.to_dict()
        log.debug(f"Calculated weights: {result}")  # 添加日志
        return weights.to_dict()