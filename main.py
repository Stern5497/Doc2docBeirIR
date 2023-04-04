import wandb
import json
import os, pathlib
import beir
from beir.datasets.data_loader import GenericDataLoader
from train_model import train
from beir import util, LoggingHandler
import ast
import random
import pandas as pd
from datasets import load_dataset
from preprocess_data import PreprocessData
from process_data import ProcessData

"""
We are using https://github.com/beir-cellar/beir/wiki/Examples-and-tutorials as an example and work with their built in 
ir models



See this colab as an example how to use ir models of beir
https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing#scrollTo=B2X_deOMFxJB

"""


def run_project():

    process_data = ProcessData()

    corpus = process_data.load_corpus("data/corpus_small.jsonl")
    qrels = process_data.load_qrels("data/qrels.jsonl")
    queries = process_data.load_queries("data/queries_facts.jsonl")
    print(queries)

    """
    corpus_splits = data['corpus'].train_test_split(test_size=0.2)
    queries_facts_splits = data['queries']['facts'].train_test_split(test_size=0.2)
    qrels_splits = data['qrels'].train_test_split(test_size=0.2)
    triplets_facts_splits = data['triplets']
    """

    # TODO find out what dev data is used for

    model_name = 'distilbert-base-uncased'
    train_loss = 'cosine'  # cosine or dot_product
    train(corpus_splits['train'], queries_facts_splits['train'], qrels_splits['train'], None, None, None, model_name=model_name, train_loss=train_loss)


if __name__ == '__main__':
    print("Start")
    run_project()
