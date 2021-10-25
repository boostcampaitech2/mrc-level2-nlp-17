import os
import json
import faiss
import pickle
import numpy as np
import pandas as pd
import random
from pprint import pprint
import GPUtil
import gc
import sys

from tqdm.auto import tqdm, trange
from typing import List, Tuple, NoReturn, Optional, Union, Callable

from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer

import torch

from datasets import (
    Dataset,
    Dataset,
)


from transformers import (
    AutoTokenizer,
    AutoConfig,
    AdamW,
    get_linear_schedule_with_warmup,
    BertModel,
    BertPreTrainedModel,
)


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]
        return pooled_output


class DenseRetrieval:
    def __init__(self, model_args, data_args, training_args) -> NoReturn:
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        if model_args.is_roberta:
            self.data_args.data_path = os.path.join(self.data_args.data_path, "roberta")

        assert not (
            data_args.pretrain_dense_encoder and data_args.use_pretrained_dense_encoder
        ), "pretrain_dense_encoder 와 use_pretrained_dense_encoder은 동시에 True가 될 수 없습니다."

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path,
        )

        self.config = AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path,
        )

        if training_args.do_predict:
            model_name_or_path = os.path.join(
                model_args.model_name_or_path,
                "dense_encoder",
            )
            self.p_encoder_path = os.path.join(model_name_or_path, "p_encoder")
            self.q_encoder_path = os.path.join(model_name_or_path, "q_encoder")
        else:
            if data_args.do_train_dense_retrieval:
                if data_args.use_pretrained_dense_encoder:
                    model_name_or_path = os.path.join(
                        training_args.output_dir,
                        "dense_encoder/pretrain",
                    )
                    self.p_encoder_path = os.path.join(model_name_or_path, "p_encoder")
                    self.q_encoder_path = os.path.join(model_name_or_path, "q_encoder")
                else:
                    model_name_or_path = model_args.model_name_or_path
                    self.p_encoder_path = model_name_or_path
                    self.q_encoder_path = model_name_or_path
            else:
                if data_args.pretrain_dense_encoder:
                    model_name_or_path = model_args.model_name_or_path
                    self.p_encoder_path = model_name_or_path
                    self.q_encoder_path = model_name_or_path
                else:
                    model_name_or_path = os.path.join(
                        training_args.output_dir, "dense_encoder"
                    )
                    self.p_encoder_path = os.path.join(model_name_or_path, "p_encoder")
                    self.q_encoder_path = os.path.join(model_name_or_path, "q_encoder")

        print("load_encoder from ", self.p_encoder_path)
        self.p_encoder = BertEncoder.from_pretrained(
            self.p_encoder_path,
            config=self.config,
        ).to(training_args.device)

        print("load_encoder from ", self.q_encoder_path)
        self.q_encoder = BertEncoder.from_pretrained(
            self.q_encoder_path,
            config=self.config,
        ).to(training_args.device)

        with open(
            os.path.join(self.data_args.data_path, self.data_args.context_path),
            "r",
            encoding="utf-8",
        ) as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        # self.ids = list(range(len(self.contexts)))

        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=self.tokenizer.tokenize,
            ngram_range=(1, 2),
            max_features=50000,
        )

        self.p_sparse_embedding = None
        self.p_dense_embedding = None
        self.indexer = None  # build_faiss()로 생성합니다.
        if data_args.do_train_dense_retrieval:
            self.get_sparse_embedding()

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        assert (
            self.p_sparse_embedding is not None
        ), "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        assert (
            self.p_dense_embedding is not None
        ), "get_dense_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[0][i]:4f}")
                print(self.contexts[doc_indices[0][i]])

            return (doc_scores, [self.contexts[doc_indices[0][i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            doc_scores, doc_indices = self.get_relevant_doc_bulk(
                query_or_dataset["question"],
                k=topk,
            )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval: ")
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

    def get_embedding(self):
        self.get_sparse_embedding()
        self.get_dense_embedding()

    def get_sparse_embedding(self):
        print("get_sparse_embedding")
        # Pickle을 저장합니다.
        pickle_name = f"sparse_embedding.bin"
        tfidfv_name = f"tfidv.bin"
        emd_path = os.path.join(self.data_args.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_args.data_path, tfidfv_name)

        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            with open(emd_path, "rb") as file:
                self.p_sparse_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build sparse passage embedding")
            self.p_sparse_embedding = self.tfidfv.fit_transform(self.contexts)
            print(self.p_sparse_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_sparse_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            print("Embedding pickle saved.")

    def tokenize_hleper(self, dataset):
        p = self.tokenizer(
            [dataset[0]],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=not self.model_args.is_roberta,
        ).to(self.training_args.device)

        temp_p = self.tokenizer(
            [dataset[1]],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=not self.model_args.is_roberta,
        ).to(self.training_args.device)

        return_dataset = {}
        for key in p.keys():
            return_dataset[key] = torch.stack((p[key], temp_p[key]), axis=0).squeeze()

        assert torch.equal(
            p["input_ids"][0], return_dataset["input_ids"][0]
        ) and torch.equal(
            temp_p["input_ids"][0], return_dataset["input_ids"][-1]
        ), "{} == {} and {} == {}".format(
            p["input_ids"],
            return_dataset["input_ids"][0],
            temp_p["input_ids"],
            return_dataset["input_ids"][-1],
        )

        for c in tqdm(dataset[2:]):
            temp_p = self.tokenizer(
                [c],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=not self.model_args.is_roberta,
            ).to(self.training_args.device)

            for key in p.keys():
                return_dataset[key] = torch.cat(
                    (return_dataset[key], temp_p[key]), axis=0
                )

            assert torch.equal(temp_p["input_ids"][0], return_dataset["input_ids"][-1])

        return return_dataset

    def get_dense_embedding(self):
        print("get_dense_embedding")
        # Pickle을 저장합니다.
        pickle_name = f"dense_embedding.bin"
        emd_path = os.path.join(self.data_args.data_path, pickle_name)

        if os.path.isfile(emd_path) and (not self.data_args.do_train_dense_retrieval):
            with open(emd_path, "rb") as file:
                self.p_dense_embedding = pickle.load(file)
                print(self.p_dense_embedding.shape)
            print("dense Embedding pickle load.")
        else:
            print("Build dense passage embedding")
            self.p_encoder.eval()
            with torch.no_grad():
                p = self.tokenizer(
                    [self.contexts[0]],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    return_token_type_ids=not self.model_args.is_roberta,
                ).to(self.training_args.device)
                p_embs1 = self.p_encoder(**p).to("cpu").numpy()

                temp_p = self.tokenizer(
                    [self.contexts[1]],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    return_token_type_ids=not self.model_args.is_roberta,
                ).to(self.training_args.device)
                temp_p_embs = self.p_encoder(**temp_p).to("cpu").numpy()

                p_embs = np.stack((p_embs1, temp_p_embs), axis=0).squeeze()

                assert np.array_equal(p_embs1[0], p_embs[0]) and np.array_equal(
                    temp_p_embs[0], p_embs[-1]
                )

                for c in tqdm(self.contexts[2:]):
                    temp_p = self.tokenizer(
                        [c],
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                        return_token_type_ids=not self.model_args.is_roberta,
                    ).to(self.training_args.device)
                    temp_p_embs = self.p_encoder(**temp_p).to("cpu").numpy()
                    p_embs = np.concatenate((p_embs, temp_p_embs), axis=0)
                    assert np.array_equal(temp_p_embs[0], p_embs[-1])

            self.p_dense_embedding = torch.Tensor(
                p_embs
            ).squeeze()  # (num_passage, emb_dim)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_dense_embedding, file)
            print("dense Embedding pickle saved.")

    def prepare_negative(
        self,
        dataset,
        p_with_n_num,
        use_tfidf_top_k_negativ,
        pretrain_dense_encoder=False,
    ):
        print("start prepare_negative")

        # 너무 데이터가 많은 경우 시간이 오르걸리므로 max_dataset_num으로 제한
        max_dataset_num = self.data_args.pretrain_max_dataset_num

        if pretrain_dense_encoder and use_tfidf_top_k_negativ:
            print("pretrain을 하는 경우 negative context를 준비할 때 tfidf를 사용하지 않음")
            use_tfidf_top_k_negativ = False

        # here
        random_negative_name = f"random_negative.bin"
        tfidf_negative_name = f"tfidf_negative.bin"
        pretrain_negative_name = f"preprocessing_for_ict_negative.bin"

        random_path = os.path.join(self.data_args.data_path, random_negative_name)
        tfidf_path = os.path.join(self.data_args.data_path, tfidf_negative_name)
        pretrain_path = os.path.join(self.data_args.data_path, pretrain_negative_name)

        if pretrain_dense_encoder:
            if os.path.isfile(pretrain_path):
                with open(pretrain_path, "rb") as file:
                    load_dataset = pickle.load(file)
                    assert len(load_dataset) == min(
                        len(dataset), max_dataset_num
                    ), "{} != {} preprocessing_for_ict_negative.bin 파일을 삭제해 주세요".format(
                        len(load_dataset), len(dataset)
                    )
                return load_dataset
        else:
            if use_tfidf_top_k_negativ:
                if os.path.isfile(tfidf_path):
                    with open(tfidf_path, "rb") as file:
                        load_dataset = pickle.load(file)
                        assert len(load_dataset) == len(
                            dataset
                        ), "{} != {} tfidf_negative.bin 파일을 삭제해 주세요".format(
                            len(load_dataset), len(dataset)
                        )
                    return load_dataset
            else:
                if os.path.isfile(random_path):
                    with open(random_path, "rb") as file:
                        load_dataset = pickle.load(file)
                        assert len(load_dataset) == len(
                            dataset
                        ), "{} != {} random_negative.bin 파일을 삭제해 주세요".format(
                            len(load_dataset), len(dataset)
                        )
                    return load_dataset

        questions = [data["question"] for data in dataset]
        p_with_n_contexts = []
        targets = []

        for i, data in tqdm(enumerate(dataset)):
            query = data["question"]
            context = data["context"]

            if use_tfidf_top_k_negativ:
                query_emb = self.tfidfv.transform([query])
                result = query_emb * self.p_sparse_embedding.T
                if not isinstance(result, np.ndarray):
                    result = result.toarray()

                sorted_result = np.argsort(result.squeeze())[::-1]
                # doc_score = result.squeeze()[sorted_result].tolist()[:p_with_n_num]
                doc_indices = sorted_result.tolist()
                negative_contexts = [
                    self.contexts[idx]
                    for idx in doc_indices
                    if self.contexts[idx] != context
                ]
                negative_contexts = negative_contexts[: p_with_n_num - 1]

            else:
                negative_contexts = [
                    c
                    for c in random.choices(self.contexts, k=p_with_n_num + 1)
                    if c != context
                ]
                negative_contexts = negative_contexts[: p_with_n_num - 1]

            assert len(negative_contexts) == (
                p_with_n_num - 1
            ), "len(negative_contexts) : {}".format(len(negative_contexts))

            assert (
                context not in negative_contexts
            ), "ground : {}, negative : {}".format(context, negative_contexts)

            negative_contexts.insert(i % p_with_n_num, context)
            targets.append(i % p_with_n_num)
            p_with_n_contexts.extend(negative_contexts)

        assert (len(questions) * p_with_n_num) == len(
            p_with_n_contexts
        ), "questions len : {}, p_with_n len : {}".format(
            len(questions), len(p_with_n_contexts)
        )

        random_question_sample_idx = random.sample(
            range(len(questions)), min(max_dataset_num, len(questions))
        )
        questions_sample = [questions[i] for i in random_question_sample_idx]
        targets = [targets[i] for i in random_question_sample_idx]
        assert len(questions_sample) == len(targets)

        random_p_with_n_sample = [i * p_with_n_num for i in random_question_sample_idx]
        p_with_n_contexts_sample = []
        for idx in random_p_with_n_sample:
            p_with_n_contexts_sample.extend(p_with_n_contexts[idx : idx + p_with_n_num])

        assert (len(questions_sample) * p_with_n_num) == len(p_with_n_contexts_sample)

        print("tokenize questions")
        q_seqs = self.tokenize_hleper(questions_sample)

        print("tokenize p_with_n_contexts")
        p_seqs = self.tokenize_hleper(p_with_n_contexts_sample)

        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, p_with_n_num, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(
            -1, p_with_n_num, max_len
        )

        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(
            -1, p_with_n_num, max_len
        )

        print("p_seqs['input_ids'].size()", p_seqs["input_ids"].size())
        print("q_seqs['input_ids'].size()", q_seqs["input_ids"].size())

        dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
            torch.tensor(targets),
        )

        print("save negative dataset")
        if use_tfidf_top_k_negativ:
            with open(tfidf_path, "wb") as file:
                pickle.dump(dataset, file)
        else:
            if pretrain_dense_encoder:
                with open(pretrain_path, "wb") as file:
                    pickle.dump(dataset, file)
            else:
                with open(random_path, "wb") as file:
                    pickle.dump(dataset, file)

        return dataset

    def pretrain(self):
        print("start pretrain dense encoder")
        ict_path = os.path.join(self.data_args.data_path, "preprocessing_for_ict.bin")

        if os.path.isfile(ict_path):
            with open(ict_path, "rb") as file:
                dataset = pickle.load(file)
            print("pretrain train : ", dataset)
            self.train(dataset, is_pretrain=True)
        else:
            print("ict data 가 없습니다.")

    def train(self, train_dataset, is_pretrain=False):
        print("start dense retrieval train")
        # 1개의 positive_context와 p_with_n_num-1개의 negative_context를 사용할 예정
        print(train_dataset)
        p_with_n_num = self.data_args.p_with_n_num

        args = self.training_args
        args.output_dir = os.path.join(args.output_dir, "dense_encoder")

        batch_size = args.per_device_train_batch_size

        no_decay = [
            "bias",
            "LayerNorm.weight",
        ]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
        )

        print("2")
        GPUtil.showUtilization()
        dataset_list = []
        dataset_list.append(
            self.prepare_negative(
                train_dataset,
                p_with_n_num,
                use_tfidf_top_k_negativ=False,
                pretrain_dense_encoder=is_pretrain,
            )
        )
        print("data_list size", sys.getsizeof(dataset_list))

        if not is_pretrain:
            dataset_list.append(
                self.prepare_negative(
                    train_dataset, p_with_n_num, use_tfidf_top_k_negativ=True
                )
            )

        prepare_train_dataset = torch.utils.data.ConcatDataset(dataset_list)

        train_dataloader = DataLoader(
            prepare_train_dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )

        t_total = len(train_dataloader)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        print("t_total", t_total)

        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        if is_pretrain:
            args.num_train_epochs = 1
        else:
            args.num_train_epochs = 3

        train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

        with torch.set_grad_enabled(True):

            self.p_encoder.train()
            self.q_encoder.train()

            for _ in train_iterator:
                t = 0
                f = 0
                with tqdm(train_dataloader, unit="batch") as tepoch:

                    for batch in tepoch:

                        p_inputs = {
                            "input_ids": batch[0]
                            .view(batch_size * (p_with_n_num), -1)
                            .to(args.device),
                            "attention_mask": batch[1]
                            .view(batch_size * (p_with_n_num), -1)
                            .to(args.device),
                            "token_type_ids": batch[2]
                            .view(batch_size * (p_with_n_num), -1)
                            .to(args.device),
                        }

                        q_inputs = {
                            "input_ids": batch[3].to(args.device),
                            "attention_mask": batch[4].to(args.device),
                            "token_type_ids": batch[5].to(args.device),
                        }

                        targets = batch[6].to(args.device)

                        del batch
                        torch.cuda.empty_cache()

                        p_outputs = self.p_encoder(**p_inputs)
                        q_outputs = self.q_encoder(**q_inputs)

                        assert p_outputs.shape[0] == (
                            batch_size * p_with_n_num
                        ), "{} == {}".format(p_outputs.shape, q_outputs.shape)

                        p_outputs = p_outputs.view(batch_size, p_with_n_num, -1)
                        p_outputs_T = torch.transpose(p_outputs, 1, 2)
                        q_outputs = q_outputs.view(batch_size, 1, -1)

                        sim_scores = torch.bmm(
                            q_outputs, p_outputs_T
                        ).squeeze()  # batch matrix multiplication

                        sim_scores = sim_scores.view(batch_size, -1)
                        sim_scores = torch.nn.functional.log_softmax(sim_scores, dim=1)
                        _, preds = torch.max(sim_scores.to("cpu"), 1)
                        loss = torch.nn.functional.nll_loss(sim_scores, targets)

                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                        self.q_encoder.zero_grad()
                        self.p_encoder.zero_grad()
                        optimizer.zero_grad()

                        global_step += 1

                        del p_inputs, q_inputs
                        torch.cuda.empty_cache()

                        local_t = int(sum(preds == targets.to("cpu")))
                        t += local_t
                        local_f = len(preds) - local_t
                        f += local_f

                        tepoch.set_postfix(
                            loss=f"{loss.item():.3f}", acc=f"{t / (t + f):.3f}"
                        )

        if is_pretrain:
            output_dir = os.path.join(args.output_dir, "pretrain")
        else:
            output_dir = args.output_dir
            self.get_dense_embedding()
            self.build_faiss()

        p_output_dir = os.path.join(output_dir, "p_encoder")
        self.p_encoder.save_pretrained(p_output_dir)

        q_output_dir = os.path.join(output_dir, "q_encoder")
        self.q_encoder.save_pretrained(q_output_dir)

    def build_faiss(self, num_clusters=64):
        indexer_name = f"dense_faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_args.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        if (
            not os.path.isfile(indexer_path)
        ) or self.data_args.do_train_dense_retrieval:

            p_emb = self.p_dense_embedding.numpy()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        torch.cuda.empty_cache()
        with torch.no_grad():
            self.q_encoder.eval()

            q_seqs_val = self.tokenizer(
                [query],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=not self.model_args.is_roberta,
            ).to(self.training_args.device)
            q_embs = self.q_encoder(**q_seqs_val).to("cpu")  # (num_query, emb_dim)

            doc_scores = []
            doc_indices = []
            print(q_embs.shape, self.p_dense_embedding.shape, "내적하는 중")
            dot_prod_scores = torch.matmul(
                q_embs, torch.transpose(self.p_dense_embedding, 0, 1)
            )
            rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

            sorted_result = rank
            doc_scores.append(dot_prod_scores[0, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])

        return doc_scores, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        torch.cuda.empty_cache()
        with torch.no_grad():
            self.q_encoder.eval()
            q = self.tokenizer(
                [queries[0]],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=not self.model_args.is_roberta,
            ).to(self.training_args.device)
            q_embs1 = self.q_encoder(**q).to("cpu").numpy()

            temp_q = self.tokenizer(
                [queries[1]],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=not self.model_args.is_roberta,
            ).to(self.training_args.device)
            temp_q_embs = self.q_encoder(**temp_q).to("cpu").numpy()
            q_embs = np.stack((q_embs1, temp_q_embs), axis=0).squeeze()

            assert np.array_equal(q_embs1[0], q_embs[0]) and np.array_equal(
                temp_q_embs[0], q_embs[-1]
            )

            for c in tqdm(queries[2:]):
                temp_q = self.tokenizer(
                    [c],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    return_token_type_ids=not self.model_args.is_roberta,
                ).to(self.training_args.device)
                temp_q_embs = self.q_encoder(**temp_q).to("cpu").numpy()
                q_embs = np.concatenate((q_embs, temp_q_embs), axis=0)
                assert np.array_equal(temp_q_embs[0], q_embs[-1])

            q_embs = torch.Tensor(q_embs).squeeze()  # (num_passage, emb_dim)

            doc_scores = []
            doc_indices = []
            print(q_embs.shape, self.p_dense_embedding.shape, "내적하는 중")
            dot_prod_scores = torch.matmul(
                q_embs, torch.transpose(self.p_dense_embedding, 0, 1)
            )
            rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

            for i in range(dot_prod_scores.shape[0]):
                sorted_result = rank[i]
                doc_scores.append(dot_prod_scores[i, :][sorted_result].tolist()[:k])
                doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(queries, k=topk)
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval: ")
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

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        torch.cuda.empty_cache()
        self.p_encoder = self.p_encoder.to("cpu")
        self.q_encoder = self.q_encoder.to("cpu")
        with torch.no_grad():
            self.p_encoder.eval()
            self.q_encoder.eval()

            q_seqs_val = self.tokenizer(
                [query],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=not self.model_args.is_roberta,
            )
            q_embs = self.q_encoder(**q_seqs_val)  # (num_query, emb_dim)

        q_embs = q_embs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()

    def get_relevant_doc_bulk_faiss(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        torch.cuda.empty_cache()
        self.p_encoder = self.p_encoder.to("cpu")
        self.q_encoder = self.q_encoder.to("cpu")
        with torch.no_grad():
            self.p_encoder.eval()
            self.q_encoder.eval()

            q_seqs_val = self.tokenizer(
                queries,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=not self.model_args.is_roberta,
            )
            q_embs = self.q_encoder(**q_seqs_val)  # (num_query, emb_dim)

        q_embs = q_embs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()
