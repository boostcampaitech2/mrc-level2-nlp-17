from retrievers.DenseRetrieval import DenseRetrieval
from retrievers.elastic_search import ElasticSearchRetrieval
from typing import List, Tuple, NoReturn, Any, Optional, Union, Dict, Callable
from datasets import Dataset

import pandas as pd


class DenseAndElasticRetrieval:
    def __init__(self, model_args, data_args, training_args):
        self.dense_retrieval = DenseRetrieval(model_args, data_args, training_args)
        self.elastic_retrieval = ElasticSearchRetrieval(
            model_args, data_args, training_args
        )

    def get_embedding(self):
        self.dense_retrieval.get_embedding()

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        dense_topk = topk // 2
        elastic_topk = topk - dense_topk

        dense_cqas = self.dense_retrieval.retrieve(query_or_dataset, dense_topk)
        elastic_cqas = self.elastic_retrieval.retrieve(query_or_dataset, elastic_topk)

        elastic_cqas.append(dense_cqas)

        return elastic_cqas
