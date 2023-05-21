import wandb
import json
import os, pathlib
import beir
from beir.datasets.data_loader import GenericDataLoader
from train_model import train
from evaluate_dpr import run_evaluate_dpr
from evaluate_sbert import run_evaluate_sbert
from evaluate_dim_reduction import run_evaluate_dim_reduction
from beir import util, LoggingHandler
import ast
import random
import pandas as pd
from datasets import load_dataset
from process_data import ProcessData

"""
We are using https://github.com/beir-cellar/beir/wiki/Examples-and-tutorials as an example and work with their built in 
ir models



See this colab as an example how to use ir models of beir
https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing#scrollTo=B2X_deOMFxJB

"""


def run_project():
    print("START")

    """
    Get Data
    """

    process_data = ProcessData()
    querie_dataset, qrel_dataset, corpus_dataset = process_data.get_data()

    data = {}

    # filter for languages
    data['corpus'] = corpus_dataset
    data['query'] = {'de': querie_dataset.filter(lambda x: x['language'] == 'de')}
    data['query']['fr'] = querie_dataset.filter(lambda x: x['language'] == 'fr')
    data['query']['it'] = querie_dataset.filter(lambda x: x['language'] == 'it')
    data['query']['mixed'] = querie_dataset.filter(lambda x: x['language'] == 'mixed')
    data['qrel'] = {'de': qrel_dataset.filter(lambda x: x['language'] == 'de')}
    data['qrel']['fr'] = qrel_dataset.filter(lambda x: x['language'] == 'fr')
    data['qrel']['it'] = qrel_dataset.filter(lambda x: x['language'] == 'it')
    data['qrel']['mixed'] = qrel_dataset.filter(lambda x: x['language'] == 'mixed')
    print("After filtering for language")
    print(data.keys())

    """
    Split Data
    """

    # create Train Test split (only for mixed dataset)
    queries, qrels, corpus, _, __, ___ = process_data.create_data_dicts(data['query']['mixed'], data['qrel']['mixed'],
                                                                        data['corpus'])
    n = 0.5
    qrels_train, queries_train, qrels_test, queries_test = process_data.create_splits(n, queries, qrels, corpus)

    """
    Train S-BERT
   

    model_name = 'joelito/legal-xlm-roberta-base'
    train_loss = 'cosine'  # cosine or dot_product
    train(corpus, queries_train, qrels_train, None, None, None, model_name=model_name, train_loss=train_loss,
          pretrained_model=True)

    model_name = 'joelito/legal-swiss-roberta-base'
    train_loss = 'cosine'  # cosine or dot_product
    train(corpus, queries_train, qrels_train, None, None, None, model_name=model_name, train_loss=train_loss,
          pretrained_model=True)
    """

    """
    Evaluate sBERT and Dim REduction
    """

    stopwords_remove = False
    shorten = False
    subset = False
    sll = False

    querie_language = 'mixed'
    qrel_language = 'mixed'

    print("Before Prozessing Test split:")
    print(f"Queries: {len(queries_test.items())}")
    print(f"Qrels: {len(qrels_test.items())}")
    print(f"Corpus: {len(corpus.items())}")

    if shorten:
        queries_test, corpus = process_data.shorten_and_reduce(queries_test, corpus, 'german')
        queries_test, corpus = process_data.shorten_and_reduce(queries_test, corpus, 'french')
        queries_test, corpus = process_data.shorten_and_reduce(queries_test, corpus, 'italian')
        print(len(queries_test.items()))

    if stopwords_remove:
        queries_test, corpus = process_data.shorten_and_reduce(queries_test, corpus, "german")
        queries_test, corpus = process_data.shorten_and_reduce(queries_test, corpus, "italian")
        queries_test, corpus = process_data.shorten_and_reduce(queries_test, corpus, "french")

    qrels_test_mixed, queries_test_mixed = process_data.create_sll(data['qrel']['mixed'], qrels_test, queries_test, corpus)
    qrels_test_de, queries_test_de = process_data.create_sll(data['qrel']['de'], qrels_test, queries_test, corpus)
    qrels_test_fr, queries_test_fr = process_data.create_sll(data['qrel']['fr'], qrels_test, queries_test, corpus)
    qrels_test_it, queries_test_it = process_data.create_sll(data['qrel']['it'], qrels_test, queries_test, corpus)

    missing_queries = 0
    for query_id, querie_text in queries_test_mixed.items():
        if not query_id in qrels_test_mixed:
            missing_queries = missing_queries+1

    print(f"{missing_queries}query missing in qrel")

    missing_qrels = 0
    missing_corp = 0
    for query_key, qrel_value in qrels_test_mixed.items():
        if not query_key in queries_test_mixed:
            missing_qrels = missing_qrels+1
            for corp_id in qrel_value:
                if not corp_id in corpus:
                    missing_corp = missing_corp + 1

    print(f"{missing_qrels}query in qrel missing in query")
    print(f"{missing_corp} corpus missing in qrel")



    print("After Prozessing Test split:")
    print(f"Queries: {len(queries_test_mixed.items())}")
    print(f"Qrels: {len(qrels_test_mixed.items())}")
    print(f"Corpus: {len(corpus.items())}")

    print("####################################################################################")
    model_name = 'distiluse-base-multilingual-cased-v1'  # multilingual model - not trained
    print(model_name)
    print("####################################################################################")
    print('dim reduction')
    run_evaluate_dim_reduction(corpus, queries_test_mixed, qrels_test_mixed, model_name)
    print("####################################################################################")
    print('SBERT')
    print('mixed')
    run_evaluate_sbert(corpus, queries_test_mixed, qrels_test_mixed, model_name)
    print("####################################################################################")
    print('de')
    run_evaluate_sbert(corpus, queries_test_de, qrels_test_de, model_name)
    print("####################################################################################")
    print('fr')
    run_evaluate_sbert(corpus, queries_test_fr, qrels_test_fr, model_name)
    print("####################################################################################")
    print('it')
    run_evaluate_sbert(corpus, queries_test_it, qrels_test_it, model_name)
    """
    print("####################################################################################")
    model_name = 'Stern5497/dummy'  # multilingual model - not trained
    print(model_name)
    print("####################################################################################")
    print('dim reduction')
    run_evaluate_dim_reduction(corpus, queries_test_mixed, queries_test_mixed, model_name)
    print("####################################################################################")
    print('SBERT')
    print('mixed')
    run_evaluate_sbert(corpus, queries_test_mixed, qrels_test_mixed, model_name)
    print("####################################################################################")
    print('de')
    run_evaluate_sbert(corpus, queries_test_de, qrels_test_de, model_name)
    print("####################################################################################")
    print('fr')
    run_evaluate_sbert(corpus, queries_test_fr, qrels_test_fr, model_name)
    print("####################################################################################")
    print('it')
    run_evaluate_sbert(corpus, queries_test_it, qrels_test_it, model_name)

    print("####################################################################################")
    model_name = 'Stern5497/sBert-swiss-legal-base'  # multilingual model - trained
    print(model_name)
    print("####################################################################################")
    print('SBERT')
    print('mixed')
    run_evaluate_sbert(corpus, queries_test_mixed, qrels_test_mixed, model_name)
    print("####################################################################################")
    print('de')
    run_evaluate_sbert(corpus, queries_test_de, qrels_test_de, model_name)
    print("####################################################################################")
    print('fr')
    run_evaluate_sbert(corpus, queries_test_fr, qrels_test_fr, model_name)
    print("####################################################################################")
    print('it')
    run_evaluate_sbert(corpus, queries_test_it, qrels_test_it, model_name)

    print("####################################################################################")
    model_name = 'Stern5497/sBert-legal-xlm-base'  # multilingual model - trained
    print(model_name)
    print("####################################################################################")
    print('SBERT')
    print('mixed')
    run_evaluate_sbert(corpus, queries_test_mixed, qrels_test_mixed, model_name)
    print("####################################################################################")
    print('de')
    run_evaluate_sbert(corpus, queries_test_de, qrels_test_de, model_name)
    print("####################################################################################")
    print('fr')
    run_evaluate_sbert(corpus, queries_test_fr, qrels_test_fr, model_name)
    print("####################################################################################")
    print('it')
    run_evaluate_sbert(corpus, queries_test_it, qrels_test_it, model_name)

    """





if __name__ == '__main__':
    print("Start")
    run_project()





