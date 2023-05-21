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

        querie_dataset = querie_dataset.filter(lambda row: len(row['text']) > 3)
        corpus_dataset = corpus_dataset.filter(lambda row: len(row['text']) > 3)

        return querie_dataset, qrel_dataset, corpus_dataset

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

    def shorten_and_reduce(self, queries, corpus, language_long):

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

        print("Shortened queries and removed stopwords")

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

    def create_splits(self, n, queries, qrels, corpus):
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

    def create_sll(self, qrels_lang_dataset, qrels_test, queries_test, _):
        counter = 0
        qrels_sll = {}
        queries_ssl = {}
        ids = []
        print(len(queries_test.items()))
        print(f"Qrels_test includes: {len(qrels_test)} before ssl removal")

        qrel = {}

        def write_ssl_qrels(row):
            # only use qrels with valid query
            if row['id'] not in qrel:
                qrel[row['id']] = {row['corp_id']: 1}
                ids.append(row['id'])
            else:
                qrel[row['id']][row['corp_id']] = 1
                ids.append(row['id'])
            return row
        qrels_lang_dataset = qrels_lang_dataset.map(write_ssl_qrels)

        missing_query = 0

        for q_key, value in qrel.items():
            # only use qrels with valid query
            if q_key in queries_test:
                for c_key in value.keys():
                    if c_key not in qrels_sll:
                        qrels_sll[q_key] = {c_key: 1}
                        ids.append(q_key)
                    else:
                        qrels_sll[q_key][c_key] = 1
                        ids.append(q_key)
            else:
                missing_query = missing_query + 1

        print(f"Missing queries: {missing_query}")
        print(f"Qrels SLL includes: {len(qrels_sll.items())}")

        query = {}

        for key, value in queries_test.items():
            if key in ids:
                query[key] = value

        return qrels_sll, query

    def load_corpus(self, corpus_files):
        corpus = {}
        for path in corpus_files:
            num_lines = sum(1 for i in open(path, 'rb'))
            with open(path, encoding='utf8') as fIn:
                for line in tqdm(fIn, total=num_lines):
                    line = json.loads(line)
                    content = line.get("content")
                    content['text'] = re.sub('\W+', ' ', content['text'])
                    corpus[line.get("id")] = content
        return corpus

    def load_queries_splits(self, query_file):
        test_queries = {}
        train_queries = {}
        val_queries = {}
        num_lines = sum(1 for i in open(query_file, 'rb'))
        test = num_lines*0.2
        val = num_lines*0.3
        counter = 0
        with open(query_file, encoding='utf8') as fIn:
            for line in fIn:
                counter += 1
                line = json.loads(line)
                if counter < test:
                    test_queries[line.get("id")] = line.get("text")
                elif counter < val:
                    val_queries[line.get("id")] = line.get("text")
                else:
                    train_queries[line.get("id")] = line.get("text")
        return train_queries, test_queries, val_queries

    def load_queries(self, query_files):
        queries = {}
        for path in query_files:
            with open(path, encoding='utf8') as fIn:
                for line in fIn:
                    line = json.loads(line)
                    text = re.sub('\W+', ' ', line.get("text"))
                    queries[line.get("id")] = text
        return queries

    def load_qrels(self, qrels_files):
        qrels = {}
        for path in qrels_files:
            with open(path, encoding='utf8') as fIn:
                for line in fIn:
                    line = json.loads(line)
                    id = line.get("id")
                    cit_id = line.get('corp_id')
                    if id not in qrels:
                        qrels[id] = {cit_id : 1}
                    else:
                        qrels[id][cit_id] = 1
        return qrels

    def load_triplets(self, triplets_file, feature='facts'):
        triplets = []
        with open(triplets_file, encoding='utf8') as fIn:
            for line in fIn:
                line = json.loads(line)
                text_feature = line.get(feature)
                text_corpus = line.get('citations')
                text_corpus_neg = line.get('neg_text')
                triplets.append((text_feature, text_corpus, text_corpus_neg))
        return triplets
