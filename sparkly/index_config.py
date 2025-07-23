from copy import deepcopy
import json
from sparkly.utils import type_check, type_check_iterable, type_check_call
from typing import Iterable, Dict, Tuple

class IndexConfig:

    @type_check_call
    def __init__(self, *, store_vectors: bool=False, id_col: str='_id', weighted_queries: bool=False):
        self.field_to_analyzers = {}# 字段到分析器的映射
        self.concat_fields = {}# 组合字段的配置
        self._id_col = id_col# 唯一标识列
        self.default_analyzer = 'standard'# 默认分析器
        self.sim = {'type' : 'BM25', 'k1' : 1.2, 'b' : .75}  # 相似度算法配置
        type_check(store_vectors, 'store_vectors', bool)  # 是否存储向量
        self._store_vectors = store_vectors
        self._frozen = False
        self._weighted_queries = weighted_queries # 是否使用权重查询
        # 新增: 字段权重存储
        self._field_weights = {}  # 字段到权重的映射
    

    def freeze(self):
        """
        Returns
        -------
        IndexConfig
            a frozen deepcopy of this index config
        """
        o = deepcopy(self)
        o._frozen = True
        return o

    @property
    def is_frozen(self):
        """
        Returns
        -------
        bool
            True if this index is frozen (not modifiable) else False
        """
        return self._frozen
    @property
    def weighted_queries(self):
        """
        True if the term vectors in the index should be stored, else False
        """
        return self._weighted_queries

    @weighted_queries.setter
    @type_check_call
    def weighted_queries(self, o: bool):
        self._raise_if_frozen()
        self._weighted_queries = o

    @property
    def store_vectors(self):
        """
        True if the term vectors in the index should be stored, else False
        """
        return self._store_vectors

    @store_vectors.setter
    @type_check_call
    def store_vectors(self, o: bool):
        self._raise_if_frozen()
        self._store_vectors = o
    
    @property
    def id_col(self):
        """
        The unique id column for the records in the index this must be a 32 or 64 bit integer
        """
        return self._id_col

    @id_col.setter
    @type_check_call
    def id_col(self, o: str):
        self._raise_if_frozen()
        self._id_col = o

    @property
    def weights(self):
        """获取属性配置的权重"""
        return self._weights

    @weights.setter
    def weights(self, weights: Dict[Tuple[str, str], float]):
        """设置属性配置的权重"""
        self._weights = weights

    @classmethod
    def from_json(cls, data):
        """
        construct an index config from a dict or json string,
        see IndexConfig.to_dict for expected format

        Returns
        -------
        IndexConfig
        """
        # """从JSON数据构造配置"""
        if isinstance(data, str):
            data = json.loads(data)

        o = cls()
        o.field_to_analyzers = data['field_to_analyzers']
        o.concat_fields = data['concat_fields']
        o.default_analyzer = data['default_analyzer']
        o.sim = data['sim']
        o.id_col = data['id_col']
        o.weighted_queries = data['weighted_queries']
        # 新增: 加载字段权重（处理向后兼容）
        if 'field_weights' in data:
            o._field_weights = data['field_weights']
        return o

    def to_dict(self):
        """
        convert this IndexConfig to a dictionary which can easily 
        be stored as json

        Returns
        -------
        dict
            A dictionary representation of this IndexConfig
        """
        d = {
                'field_to_analyzers' : self.field_to_analyzers,
                'concat_fields' : self.concat_fields,
                'default_analyzer' : self.default_analyzer,
                'sim' : self.sim,
                'store_vectors' : self.store_vectors,
                'id_col' : self.id_col,
                'weighted_queries' : self.weighted_queries,
                # 新增: 保存字段权重
                'field_weights': self._field_weights
        }
        return d

    def to_json(self):
        """
        Dump this IndexConfig to a valid json strings

        Returns
        -------
        str
        """
        return json.dumps(self.to_dict())

    @type_check_call
    def add_field(self, field : str, analyzers: Iterable[str]):
        """
        Add a new field to be indexed with this config

        Parameters
        ----------

        field : str
            The name of the field in the table to the index （添加一个字段到索引中）
            例如：add_field('name', ['standard', '3gram'])
            会将 'name' 字段添加到索引中，并使用两种分析器处理
        analyzers : set, list or tuple of str
            The names of the analyzers that will be used to index the field
        """
        self._raise_if_frozen()
        self.field_to_analyzers[field] = list(analyzers)

        return self

    @type_check_call
    def remove_field(self, field: str):
        """
        remove a field from the config

        Parameters
        ----------

        field : str 
            the field to be removed from the config

        Returns
        -------
        bool 
            True if the field existed else False
        """

        self._raise_if_frozen()
        if field in self.field_to_analyzers:
            self.field_to_analyzers.pop(field)
            if field in self.concat_fields:
                self.concat_fields.pop(field)
            return True
        else:
            return False

    @type_check_call
    def add_concat_field(self, field : str, concat_fields: Iterable[str], analyzers: Iterable[str]):
        """
        Add a new concat field to be indexed with this config

        Parameters
        ----------

        field : str
            The name of the field that will be added to the index

        concat_fields : set, list or tuple of strs
            the fields in the table that will be concatenated together to create `field`
        会创建一个新的组合字段，将name和address的内容合并,并使用相应的分析器进行分词
        analyzers : set, list or tuple of str
            The names of the analyzers that will be used to index the field
        """
        self._raise_if_frozen()
        self.concat_fields[field] = list(concat_fields)
        self.field_to_analyzers[field] = list(analyzers)

        return self

    def get_analyzed_fields(self, query_spec=None):
        """
        Get the fields used by the index or query_spec. If `query_spec` is None, 
        the fields that are used by the index are returned.

        Parameters
        ----------

        query_spec : QuerySpec, optional
            if provided, the fields that are used by `query_spec` in creating a query

        Returns
        -------
        list of str
            the fields used
        """
        if query_spec is not None:
            fields = []
            for f in query_spec:
                if f in self.concat_fields:
                    fields += self.concat_fields[f]
                else:
                    fields.append(f)
        else:
            fields = sum(self.concat_fields.values(), [])
            fields += (x for x in self.field_to_analyzers if x not in self.concat_fields) 

        return list(set(fields))

    def _raise_if_frozen(self):
        if self.is_frozen:
            raise RuntimeError('Frozen IndexConfigs cannot be modified')
    
    # 添加权重相关的属性和方法
    @property
    def field_weights(self):
        """获取字段权重配置"""
        return self._field_weights
    
    @field_weights.setter
    @type_check_call
    def field_weights(self, weights: Dict[str, float]):
        """设置字段权重配置
        
        Parameters
        ----------
        weights : Dict[str, float]
            字段到权重的映射字典，权重应该是正数
        """
        self._raise_if_frozen()
        # 验证权重的有效性
        for field, weight in weights.items():
            if weight < 0:
                raise ValueError(f"Field weight must be non-negative (got {weight} for {field})")
        self._field_weights = weights

    def set_field_weight(self, field: str, weight: float):
        """设置单个字段的权重
        
        Parameters
        ----------
        field : str
            字段名
        weight : float
            权重值，应该是正数
        """
        self._raise_if_frozen()
        if weight < 0:
            raise ValueError(f"Field weight must be non-negative (got {weight})")
        self._field_weights[field] = weight
    
    def normalize_weights(self):
        """归一化字段权重，使所有权重和为1"""
        self._raise_if_frozen()
        if not self._field_weights:
            return
            
        total = sum(self._field_weights.values())
        if total > 0:  # 避免除零
            self._field_weights = {
                field: weight/total 
                for field, weight in self._field_weights.items()
            }

    def unfreeze(self):
        """
        暂时解冻配置以允许修改
        Returns
        -------
        self : 返回配置自身以支持链式调用
        """
        self._frozen = False
        return self