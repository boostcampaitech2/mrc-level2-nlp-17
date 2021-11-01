from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="klue/bert-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models  klue/roberta-large klue/bert-base"
        },
    )

    is_roberta: bool = field(default=False, metadata={"help": "use Reberta model"})

    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )

    retrieval_model: str = field(
        default="ElasticSearch",
        metadata={
            "help": "Using 'SparseRetrieval', 'DenseRetrieval', 'ElasticSearch', 'DenseAndElasticRetrieval' for retieval"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_path: Optional[str] = field(
        default="../data",
        metadata={"help": "data path"},
    )

    context_path: Optional[str] = field(
        default="wikipedia_documents.json",
        metadata={"help": "data path"},
    )

    dataset_name: Optional[str] = field(
        default="../data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=10,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )
    do_retrieval_example: bool = field(
        default=False, metadata={"help": "To Show retrieval example"}
    )
    pretrain_dense_encoder: bool = field(
        default=False, metadata={"help": "whether to pretrain dense encoder"}
    )

    do_train_dense_retrieval: bool = field(
        default=False, metadata={"help": "To Train Dense retrieval"}
    )

    use_pretrained_dense_encoder: bool = field(
        default=False, metadata={"help": "use pretrained dense encoder"}
    )

    p_with_n_num: int = field(
        default=20,
        metadata={
            "help": "When Train Dense retrieval, input positive and negative passage num per one question"
        },
    )
    pretrain_max_dataset_num: int = field(
        default=10000,
        metadata={"help": "Dense Retrieval를 pretrain할 때 사용할 ict dataset의 question의 갯수"},
    )

    do_preprocessing: bool = field(
        default=False,
        metadata={"help": "Whether to remove other characters in preprocessing"},
    )

    do_postprocessing: bool = field(
        default=False,
        metadata={"help": "Whether to remove ending pos starting with J in postprocessing"},
    )
