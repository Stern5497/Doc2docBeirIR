import random

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


class ProcessData:

    def __init__(self):
        self.counter = 0

    def load_corpus(self, corpus_file):
        corpus = {}
        num_lines = sum(1 for i in open(self.corpus_file, 'rb'))
        with open(corpus_file, encoding='utf8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                corpus[line.get("id")] = line.get("content")
        return corpus

    def load_queries(self, query_file):
        queries = {}
        with open(query_file, encoding='utf8') as fIn:
            for line in fIn:
                line = json.loads(line)
                queries[line.get("id")] = line.get("text")

    def load_qrels(self, qrels_file):
        qrels = {}
        with open(qrels_file, encoding='utf8') as fIn:
            for line in fIn:
                line = json.loads(line)
                id = line.get("decision_id")
                cit_id = line.get('citations')
                if id not in qrels:
                    qrels[id] = {cit_id : 1}
                else:
                    qrels[id][cit_id] = 1

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
