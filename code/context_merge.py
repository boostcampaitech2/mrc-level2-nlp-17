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


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def top_k_context_merger(retrieval, dataset, top_k):
    f = Features(
        {
            "title": Value(dtype="string", id=None),
            "answers": Sequence(
                feature={
                    "text": Value(dtype="string", id=None),
                    "answer_start": Value(dtype="int32", id=None),
                },
                length=-1,
                id=None,
            ),
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
        }
    )

    total = []
    answers = []
    for data in tqdm(dataset):
        with HiddenPrints():
            _, negative_context_list = retrieval.retrieve(data["question"], top_k + 1)
        positive_context = data["context"]
        answer_start = data["answers"]["answer_start"][0]
        for i in range(top_k + 1):
            context_list = negative_context_list
            context_list.insert(i, positive_context)

            start = answer_start
            for j in range(i):
                start += len(context_list[j])
            data["answers"]["answer_start"][0] = start

            tmp = {
                "question": data["question"],
                "id": data["id"] + "-merge-{}".format(i),
                "context": "".join(context_list),
                "answers": data["answers"],
                "title": data["title"],
            }
            total.append(tmp)
    print(total[0])
    df = pd.DataFrame(total)
    print(df.iloc[0])
    return Dataset.from_pandas(df, features=f)


if __name__ == "__main__":

    set_seed(42)

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    datasets = load_from_disk(data_args.dataset_name)

    train_dataset = datasets["train"]

    retrieval = retrieval.get_retriever(model_args, data_args, training_args)

    dataset_list = [
        train_dataset,
        top_k_context_merger(retrieval, train_dataset, 1),
        # top_k_context_merger(retrieval, train_dataset, 2),
        # top_k_context_merger(retrieval, train_dataset, 3),
    ]

    merged_dataset = concatenate_datasets(dataset_list)

    merged_path = os.path.join(data_args.dataset_name, "merged_context")
    # merged_dataset.save_to_disk(merged_path)
