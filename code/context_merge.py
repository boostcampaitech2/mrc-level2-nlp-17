from datasets import (
    load_from_disk,
    concatenate_datasets,
    Sequence,
    Value,
    Features,
    Dataset,
)
from transformers import HfArgumentParser, TrainingArguments, set_seed

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

import retrieval
import os, sys

import pandas as pd

from tqdm import tqdm

import copy


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def top_k_context_merger(retrieval, dataset, top_k):
    f = dataset.features
    total = []
    for data in tqdm(dataset):
        with HiddenPrints():
            _, negative_context_list = retrieval.retrieve(data["question"], top_k + 1)
        positive_context = data["context"]
        negative_context_list = [negative_context for negative_context in negative_context_list if not negative_context in positive_context]
        negative_context_list = negative_context_list[:top_k]
        
        answer_start = data["answers"]["answer_start"][0]
        for i in range(top_k + 1):
            context_list = copy.deepcopy(negative_context_list)
            context_list.insert(i, positive_context)

            start = answer_start
            for j in range(i):
                start += len(context_list[j])
            data["answers"]["answer_start"][0] = start

            tmp = {
                "title": data["title"],
                "answers": data["answers"],
                "context": "".join(context_list),
                "id": data["id"] + "-merge-{}".format(i),
                "question": data["question"],
                "__index_level_0__": -1,
                "document_id": -1,
            }
            total.append(tmp)
    

    df = pd.DataFrame(total)
    return Dataset.from_pandas(df, features=f)


if __name__ == "__main__":

    set_seed(42)

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    merged_path = os.path.join(data_args.dataset_name, "merged_context")

    datasets = load_from_disk(data_args.dataset_name)

    train_dataset = datasets["train"]

    # ValueError: Datasets should ALL come from memory, or should ALL come from disk. 오류 해결
    total = []
    for data in train_dataset:
        tmp = {key: data[key] for key in data.keys()}
        total.append(tmp)
    df = pd.DataFrame(total)
    train_dataset = Dataset.from_pandas(df, features=train_dataset.features)

    retrieval = retrieval.get_retriever(model_args, data_args, training_args)

    dataset_list = [train_dataset]

    for k in range(1, data_args.merge_context_num + 1):
        dataset_list.append(top_k_context_merger(retrieval, train_dataset, k))

    merged_dataset = concatenate_datasets(dataset_list)

    datasets["train"] = merged_dataset
    datasets.save_to_disk(merged_path)
