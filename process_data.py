import random
import re

import datasets
import json
import gc
import pandas as pd
import ast
from datasets import load_dataset
from tqdm import tqdm
import wandb
import json
import os, pathlib
import re
import nltk
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


class ProcessData:

    def __init__(self):
        self.counter = 0

    def load_from_hf(self, name):
        dataset = load_dataset(name)
        dataset = dataset['train']
        return dataset

    def get_data(self):
        qrel_dataset = self.load_from_hf('Stern5497/qrel')
        querie_dataset = self.load_from_hf("Stern5497/querie")
        corpus_dataset = self.load_from_hf("Stern5497/corpus")
        querie_dataset_train, querie_dataset_test = self.create_splits(querie_dataset)
        return querie_dataset, querie_dataset_train, querie_dataset_test, qrel_dataset, corpus_dataset

    def create_splits(self, querie_dataset):
        splits = querie_dataset.train_test_split(test_size=0.9)
        querie_dataset_train = splits['train']
        querie_dataset_test = splits['test']
        return querie_dataset_train, querie_dataset_test

    def create_corpus_dict(self, corpus_dataset):
        corpus_dict = {}
        def write_dict(row):
            id = str(row['id'])
            corpus_dict[id] = {"title": '', "text": row['text']}
            return row

        corpus_dataset.apply(write_dict, axis="columns")

        return corpus_dict

    def create_qrels_dict(self, qrels_dataset, corpus_dict):
        qrels_dict = {}

        def write_qrels(row):
            if row['corp_id'] in corpus_dict:
                if row['id'] not in qrels_dict:
                    qrels_dict[row['id']] = {row['corp_id']: 1}
                else:
                    qrels_dict[row['id']][row['corp_id']] = 1

        qrel_dataset = qrels_dataset.apply(write_qrels, axis='columns')

        return qrels_dict

    def create_query_dict(self, queries_dataset, qrels_dict):
        queries_dict = {}

        def write_dict(row):
            id = row['id']
            if id in qrels_dict:
                text = row['text']
                queries_dict[id] = str(text)

        queries_dataset.apply(write_dict, axis="columns")
        return queries_dict

    def create_data_dicts(self, querie_dataset, qrel_dataset, corpus_dataset):
        corpu = {}

        def write_corpus(row):
            corpu[row['id']] = {'text': row['text'], 'title': ""}
            return row

        corpus_dataset = corpus_dataset.map(write_corpus)
        print(f"Corpus: {len(corpu.items())}")

        querie_tmp = {}

        def write_queries(row):
            id = row['id']
            text = row['text']
            querie_tmp[id] = str(text)
            return row

        querie_dataset = querie_dataset.map(write_queries, keep_in_memory=True)
        print(f"Query before cleaning: {len(querie_tmp.items())}")

        qrel = {}
        ids = []
        def write_qrels(row):
            # only use qrels with valid query
            if row['id'] in querie_tmp:
                if row['id'] not in qrel:
                    qrel[row['id']] = {row['corp_id']: 1}
                    ids.append(row['id'])
                else:
                    qrel[row['id']][row['corp_id']] = 1
                    ids.append(row['id'])
            return row
        qrel_dataset = qrel_dataset.map(write_qrels)
        print(f"Qrels: {len(qrel.items())}")

        query = {}

        for key, value in querie_tmp.items():
            if key in ids:
                query[key] = value

        print(f"Query: {len(query.items())}")

        return query, qrel, corpu, querie_dataset, qrel_dataset, corpus_dataset

    def remove_stopwords(self, queries, corpus, language_long):

        sw_nltk = stopwords.words(language_long)
        print(sw_nltk)

        queries_filtered = {}

        for key, value in queries.items():
            value = re.sub('\W+', ' ', str(value))
            words = [word for word in value.split() if word.lower() not in sw_nltk]
            new_value = " ".join(words)
            # new_value = new_value[:500]
            queries_filtered[key] = new_value

        corpus_filtered = {}

        for key, value in corpus.items():
            text = re.sub('\W+', ' ', str(value['text']))
            words = [word for word in text.split() if word.lower() not in sw_nltk]
            new_value = " ".join(words)
            # new_value = new_value[:500]
            value['text'] = new_value
            corpus_filtered[key] = value

        print("Removed stopwords")

        return queries_filtered, corpus_filtered

    def create_subset(self, n, queries, qrels, corpus):
        print("Start creating subsets")
        counter = 0
        qrels_subset = {}
        queries_subset = {}
        corpus_subset = {}

        print(len(queries.items()))
        print(len(qrels.items()))
        print(len(corpus.items()))
        short = 0

        for key, value in qrels.items():
            for corp_id, _ in value.items():
                found = False
                if len(corpus[corp_id]['text']) > 10:
                    found = True
                    corpus_subset[corp_id] = corpus[corp_id]
                else:
                    short = short + 1
            if found:
                qrels_subset[key] = value
                queries_subset[key] = queries[key]
            counter += 1
            if counter > n:
                break

        print(f"create subset of {n} queries.")
        print(f"Found {short} entries in corpus with short text.")

        return queries_subset, qrels_subset, corpus_subset

    def create_splits_old(self, n, queries, qrels, corpus):
        print("Start creating splits")
        counter = 0
        qrels_train = {}
        queries_train = {}
        qrels_test = {}
        queries_test = {}

        print(f"We have {len(queries)} queries in total. {n * len(queries)} will be used for train, the rest for test.")
        print(f"We have {len(qrels)} qrels")
        print(f"We have {len(corpus)} corpus")
        missing_corpus = 0
        missing_qrel_train = 0
        missing_qrel_test = 0

        for key, value in queries.items():
            if counter < n * len(queries):
                if key in qrels:
                    # assert all corp_ids exist in corpus!
                    queries_train[key] = value
                    qrels_train[key] = qrels[key]
                    counter = counter + 1
                    for corp_id in qrels[key].keys():
                        if corp_id not in corpus:
                            missing_corpus = missing_corpus + 1
                else:
                    missing_qrel_train = missing_qrel_train + 1
            elif counter >= n * len(queries):
                if key in qrels:
                    # assert all corp_ids exist in corpus!
                    queries_test[key] = value
                    qrels_test[key] = qrels[key]
                    counter = counter + 1
                    for corp_id in qrels[key].keys():
                        if corp_id not in corpus:
                            missing_corpus = missing_corpus + 1
                else:
                    missing_qrel_test = missing_qrel_test + 1

        print(f"created splits.")
        print(f"Found {missing_corpus} missing entries in corpus.")
        print(f"We have {len(qrels_train)} train qrels")
        print(f"We have {len(qrels_test)} test qrels")
        print(f"We have {len(queries_train)} train queries")
        print(f"We have {len(queries_test)} test queries")
        return qrels_train, queries_train, qrels_test, queries_test

