import time

from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union, Dict, Callable

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

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

from transformers import HfArgumentParser, TrainingArguments, AutoTokenizer, set_seed

import wandb
import datetime
from dateutil.tz import gettz

from retrievers.sparse import SparseRetrieval
from retrievers.elastic_search import ElasticSearchRetrieval


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


def get_retriever(
    retrieval_model: str,
    tokenize_fn: Callable[[str], List[str]],
    data_path: str,
    context_path: str,
):

    retriever_dict = {
        "SparseRetrieval": SparseRetrieval,
        "ElasticSearch": ElasticSearchRetrieval,
    }
    return retriever_dict[retrieval_model](tokenize_fn, data_path, context_path)


def run_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    model_args: ModelArguments,
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = get_retriever(
        retrieval_model=model_args.retrieval_model,
        tokenize_fn=tokenize_fn,
        data_path=data_path,
        context_path=context_path,
    )

    print("=" * 100)
    print(retriever)
    print("=" * 100)

    retriever.get_embedding()

    if "build_faiss" in dir(retriever) and data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(datasets, topk=data_args.top_k_retrieval)
    else:
        df = retriever.retrieve(datasets, topk=data_args.top_k_retrieval)

    if training_args.do_eval:
        f = Features(
            {
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
    datasets = Dataset.from_pandas(df, features=f)
    return datasets


def get_retrieval_accuracy(before_dataset, after_dataset):
    assert len(before_dataset) == len(after_dataset)

    before_dataset = before_dataset.sort("id")
    after_dataset = after_dataset.sort("id")

    t = 0
    f = 0
    for i in range(len(before_dataset)):
        if before_dataset[i]["context"] in after_dataset[i]["context"]:
            t += 1
        else:
            f += 1

    print("t: {} f: {}".format(t, f))
    assert (t + f) == len(after_dataset)
    return t / (t + f)


def eval_retrieval(model_args, data_args, training_args, datasets):

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=True,
    )

    before_dataset = datasets["train"]

    after_dataset = run_retrieval(
        tokenize_fn=tokenizer.tokenize,
        datasets=before_dataset,
        training_args=training_args,
        data_args=data_args,
        model_args=model_args,
    )
    print(
        'dataset : "train", top-k : {}, use_faiss : {}'.format(
            data_args.top_k_retrieval, data_args.use_faiss
        )
    )
    train_accuracy = get_retrieval_accuracy(before_dataset, after_dataset)
    print(
        "retrieval_accuracy :",
        train_accuracy,
    )

    before_dataset = datasets["validation"]

    after_dataset = run_retrieval(
        tokenize_fn=tokenizer.tokenize,
        datasets=before_dataset,
        training_args=training_args,
        data_args=data_args,
        model_args=model_args,
    )
    print(
        'dataset : "validation", top-k : {}, use_faiss : {}'.format(
            data_args.top_k_retrieval, data_args.use_faiss
        )
    )
    val_accuracy = get_retrieval_accuracy(before_dataset, after_dataset)
    print(
        "retrieval_accuracy :",
        val_accuracy,
    )
    wandb.log({"Train/accuracy": train_accuracy, "Val/accuracy": val_accuracy})


if __name__ == "__main__":

    set_seed(42)

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    org_dataset = load_from_disk(data_args.dataset_name)

    if training_args.do_eval:
        # wandb
        wandb.init(project="mrc-level2-nlp-retriever")

        # 파라미터 초기화
        wandb.config.update(model_args)
        wandb.config.update(data_args)
        # wandb.config.update(training_args)

        # Retriever | 21-10-01 00:00 | Retriever Model Name
        wandb.run.name = (
            "Retriever | "
            + datetime.datetime.now(gettz("Asia/Seoul")).strftime("%y-%m-%d %H:%M")
            + " | "
            + model_args.retrieval_model
        )
        wandb.run.save()

        eval_retrieval(model_args, data_args, training_args, org_dataset)

    if data_args.do_retrieval_example:
        full_ds = concatenate_datasets(
            [
                org_dataset["train"].flatten_indices(),
                org_dataset["validation"].flatten_indices(),
            ]
        )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
        print("*" * 40, "query dataset", "*" * 40)
        print(full_ds)

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=False,
        )

        retriever = get_retriever(
            retrieval_model=model_args.retrieval_model,
            tokenize_fn=tokenizer.tokenize,
            data_path=data_args.data_path,
            context_path=data_args.context_path,
        )

        retriever.get_embedding()

        query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

        if data_args.use_faiss:

            # test single query
            with timer("single query by faiss"):
                scores, indices = retriever.retrieve_faiss(query)

            # test bulk
            with timer("bulk query by exhaustive search"):
                df = retriever.retrieve_faiss(full_ds)
                df["correct"] = df["original_context"] == df["context"]

                print(
                    "correct retrieval result by faiss", df["correct"].sum() / len(df)
                )

        else:
            with timer("bulk query by exhaustive search"):
                df = retriever.retrieve(full_ds)
                df["correct"] = df["original_context"] == df["context"]
                print(
                    "correct retrieval result by exhaustive search",
                    df["correct"].sum() / len(df),
                )

            with timer("single query by exhaustive search"):
                scores, indices = retriever.retrieve(query)
