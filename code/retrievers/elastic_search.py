import sys

sys.path.append("../")

import os
import json
import time
import faiss
import pickle
import numpy as np
import pandas as pd
import re

from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union, Dict, Callable
import pprint

from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
    load_metric,
    Sequence,
    Value,
    Features,
    Dataset,
    DatasetDict,
)

from elasticsearch import Elasticsearch


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class ElasticSearchRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        try:
            self.es.transport.close()
        except:
            pass
        self.es = Elasticsearch()

        pprint.pprint(self.es.info())

        self.INDEX_NAME = "wiki_index"

    def get_embedding(self) -> NoReturn:
        # 인덱싱
        pass

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Elastic Search retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        # 쿼리 검색시 특수문자 앞에 '\' 기호 추가
        patten = """[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]"""
        query = re.sub(patten, lambda m: f"\{m.group()}", query)

        with timer("searching.."):
            result = self.es.search(index=self.INDEX_NAME, q=query, size=k)

        doc_score = [float(res["_score"]) for res in result["hits"]["hits"][:k]]
        doc_indices = [int(res["_id"]) for res in result["hits"]["hits"][:k]]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        results = []

        for query in queries:
            # 쿼리 검색시 특수문자 앞에 '\' 기호 추가
            patten = """[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]"""
            query = re.sub(patten, lambda m: f"\{m.group()}", query)

            results.append(self.es.search(index=self.INDEX_NAME, q=query, size=k))

        doc_scores = []
        doc_indices = []
        for result in results:
            doc_scores.append(
                [float(res["_score"]) for res in result["hits"]["hits"][:k]]
            )
            doc_indices.append([int(res["_id"]) for res in result["hits"]["hits"][:k]])
        return doc_scores, doc_indices
