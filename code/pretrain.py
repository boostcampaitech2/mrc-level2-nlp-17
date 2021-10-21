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

from transformers import (
    set_seed,
    HfArgumentParser,
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
)
from datasets import load_from_disk
from arguments import DataTrainingArguments, ModelArguments


def pretrain_dense_encoder(model_args, data_args, training_args):
    print("Pretrain dense encoder base on the {}".format(model_args.model_name_or_path))

    tokenizer = AutoTokenizer.pretrained_from(model_args.model_name_or_path)

    p_encoder = AutoTokenizer.pretrained_from(model_args.model_name_or_path)
    q_encoder = AutoTokenizer.pretrained_from(model_args.model_name_or_path)


if __name__ == "__main__":

    set_seed(42)
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    with open(
        os.path.join(data_args.data_path, data_args.context_path), "r", encoding="utf-8"
    ) as f:
        wiki = json.load(f)

    wiki_contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))

    if data_args.pretrain_dense_encoder:
        pretrain_dense_encoder(model_args, data_args, training_args)

    print("clear")
