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

import torch
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
            "context_id": Value(dtype="int64", id=None),
        }
    )
    wiki_path = os.path.join(data_args.data_path, "preprocessing_for_ict.bin")
    checkpoint_wiki_path = os.path.join(
        data_args.data_path, "checkpoint_preprocessing_for_ict.bin"
    )

    last_check_id = 0
    check_step = 3000
    check_time = 1
    checkpoint_dataset = None
    if os.path.isfile(checkpoint_wiki_path):
        with open(checkpoint_wiki_path, "rb") as file:
            checkpoint_dataset = pickle.load(file)
            last_check_id = checkpoint_dataset[-1]["context_id"] + 1
            print("last_check : ", checkpoint_dataset[-1]["context_id"])
            print("next id :", wiki[last_check_id]["id"])

    wiki = [wiki[i] for i in range(last_check_id, len(wiki))]

    total = []
    i = 0
    for data in tqdm(wiki):
        context = data["context"]
        id = data["id"]
        if not context:
            continue
        try:
            splited_sent = kss.split_sentences(context)
        except IndexError:
            continue
        else:
            for sent in splited_sent:
                tmp = {"question": sent, "context": context, "context_id": id}
                total.append(tmp)
        if i == (check_step * check_time):
            check_time += 1
            assert total
            dataset_dataframe = pd.DataFrame(total)
            dataset_dataframe.drop_duplicates(
                subset=["question", "context", "context_id"], inplace=True
            )
            dataset = Dataset.from_pandas(dataset_dataframe, features=f)

            if checkpoint_dataset:
                dataset = torch.utils.data.ConcatDataset([checkpoint_dataset, dataset])

            print("write", checkpoint_wiki_path)
            with open(checkpoint_wiki_path, "wb") as file:
                pickle.dump(dataset, file)
        i += 1
    assert total
    dataset_dataframe = pd.DataFrame(total)
    dataset_dataframe.drop_duplicates(subset=["question", "context"], inplace=True)
    dataset = Dataset.from_pandas(dataset_dataframe, features=f)

    if checkpoint_dataset:
        dataset = torch.utils.data.ConcatDataset([checkpoint_dataset, dataset])

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
        context_list.append((w["text"], w["document_id"]))

    context_list = sorted(context_list, key=lambda x: x[1])
    contexts, ids = zip(*context_list)

    temp = {"context": contexts, "id": ids}
    wiki = Dataset.from_dict(temp)

    print(wiki)

    preprocessing_for_ict(data_args, wiki)

    print("clear")
