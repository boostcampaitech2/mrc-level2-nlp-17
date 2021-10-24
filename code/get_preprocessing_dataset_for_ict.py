import os
import json
import time
import faiss
import pickle
import numpy as np
import pandas as pd
import re
import random
from pprint import pprint
import kss

from transformers import (
    set_seed,
    HfArgumentParser,
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
)
from datasets import load_from_disk, Dataset, Features, Sequence, Value
from arguments import DataTrainingArguments, ModelArguments

from tqdm import tqdm


def preprocessing_for_ict(data_args, wiki):
    # preprocessing_for_ict한 dataset을 pd.DataFrame으로 반환합니다.
    print("start preprocessing_for_ict")

    f = Features(
        {
            "context": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
        }
    )
    wiki_path = os.path.join(data_args.data_path, "preprocessing_for_ict.bin")

    total = []

    for data in tqdm(wiki):
        context = data["context"]
        context = context.replace("　", " ")

        if not context:
            continue

        splited_sent = kss.split_sentences(context)
        for sent in splited_sent:
            tmp = {"question": sent, "context": context}
            total.append(tmp)
    assert total
    dataset_dataframe = pd.DataFrame(total)
    dataset_dataframe.drop_duplicates(subset=["question", "context"], inplace=True)
    dataset = Dataset.from_pandas(dataset_dataframe, features=f)

    print("write", wiki_path)
    with open(wiki_path, "wb") as file:
        pickle.dump(dataset, file)


if __name__ == "__main__":

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    set_seed(42)
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    with open(
        os.path.join(data_args.data_path, data_args.context_path), "r", encoding="utf-8"
    ) as f:
        wiki = json.load(f)

    context_list = []
    for w in wiki.values():
        context_list.append(w["text"])
    temp = {"context": context_list}
    wiki = Dataset.from_dict(temp)

    preprocessing_for_ict(data_args, wiki)

    print("clear")
