import wandb
import json
import os, pathlib
import beir
from beir.datasets.data_loader import GenericDataLoader
from train_model import train
from evaluate_dpr import run_evaluate_dpr
from evaluate_sbert import run_evaluate_sbert
from evaluate_dim_reduction import run_evaluate_dim_reduction
from train_sbert_hardneg import run_train_sbert_hardneg
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

def get_formatted_data(filter_language):
    process_data = ProcessData()
    querie_dataset, querie_dataset_train, querie_dataset_test, qrels_dataset, corpus_dataset = process_data.get_data()

    print(querie_dataset_train)
    print(querie_dataset_test)
    print(qrels_dataset)
    print(corpus_dataset)

    # corpus stays always the same
    corpus_dict = process_data.create_corpus_dict(pd.DataFrame(corpus_dataset))

    # qrels are split into mixed, and single languages
    qrels_dataset_ssl = qrels_dataset.filter(lambda row: row['language'] == filter_language)
    qrels_dict = process_data.create_qrels_dict(pd.DataFrame(qrels_dataset), corpus_dict)
    qrels_dict_ssl = process_data.create_qrels_dict(pd.DataFrame(qrels_dataset_ssl), corpus_dict)

    # we always split so results are comparable
    if filter_language != 'mixed':
        querie_dataset_test = querie_dataset_test.filter(lambda row: row['language'] == filter_language)
    queries_dict_train = process_data.create_query_dict(pd.DataFrame(querie_dataset_train), qrels_dict)
    queries_dict_test = process_data.create_query_dict(pd.DataFrame(querie_dataset_test), qrels_dict)

    print("####################################################################################################")
    print("Successfully loaded data.")
    print("####################################################################################################")

    print(len(corpus_dict))
    print(len(qrels_dict))
    print(len(qrels_dict_ssl))
    print(len(queries_dict_train))
    print(len(queries_dict_test))

    return queries_dict_train, queries_dict_test, qrels_dict, qrels_dict_ssl, corpus_dict

def run_testing():
    print("####################################################################################")
    print("####################################################################################")
    ssl_queries = []
    qrels = []
    lanugages = ['de', 'fr', 'it']
    for language in lanugages:
        queries_dict_train, queries_dict_test, qrels_dict, qrels_dict_ssl, corpus_dict = get_formatted_data(language)
        # to create ssl datatset of all languages we need to celloect all qrels
        ssl_queries.append(qrels_dict_ssl)
        # make sure queries are in qrels
        query_adapted = {}
        qrels.append(qrels_dict)
        for id in queries_dict_test.keys():
            if id in qrels_dict_ssl:
                query_adapted[id] = queries_dict_test[id]

        """
        print("####################################################################################")
        print(f"queries for language {language}: {len(queries_dict_test)}")
        print("####################################################################################")
        print(language)
        model_name = 'Stern5497/sBert-swiss-legal-base'
        print(model_name)
        print("####################################################################################")
        print('SBERT')
        run_evaluate_sbert(corpus_dict, queries_dict_test, qrels_dict, model_name)
        print("####################################################################################")
        print('SSL')
        run_evaluate_sbert(corpus_dict, query_adapted, qrels_dict_ssl, model_name)
        print("####################################################################################")

        print("####################################################################################")
        print(language)
        model_name = 'distiluse-base-multilingual-cased-v1'
        print(model_name)
        print("####################################################################################")
        print('SBERT')
        run_evaluate_sbert(corpus_dict, queries_dict_test, qrels_dict, model_name)
        print("####################################################################################")
        print('SSL')
        run_evaluate_sbert(corpus_dict, query_adapted, qrels_dict_ssl, model_name)
        print("####################################################################################")

        print("####################################################################################")
        print(language)
        model_name = 'Stern5497/sbert-legal-swiss-roberta-base'
        print(model_name)
        print("####################################################################################")
        print('SBERT')
        run_evaluate_sbert(corpus_dict, queries_dict_test, qrels_dict, model_name)
        print("####################################################################################")
        print('SSL')
        run_evaluate_sbert(corpus_dict, query_adapted, qrels_dict_ssl, model_name)
        print("####################################################################################")


        print("####################################################################################")
        print(language)
        model_name = 'Stern5497/sbert-distiluse'
        print(model_name)
        print("####################################################################################")
        print('SBERT')
        run_evaluate_sbert(corpus_dict, queries_dict_test, qrels_dict, model_name)
        print("####################################################################################")
        print('SSL')
        run_evaluate_sbert(corpus_dict, query_adapted, qrels_dict_ssl, model_name)
        print("####################################################################################")


        print("####################################################################################")
        print(language)
        model_name = 'Stern5497/sbert-distiluse-hardneg'
        print(model_name)
        print("####################################################################################")
        print('SBERT')
        run_evaluate_sbert(corpus_dict, queries_dict_test, qrels_dict, model_name)
        print("####################################################################################")
        print('SSL')
        run_evaluate_sbert(corpus_dict, query_adapted, qrels_dict_ssl, model_name)
        print("####################################################################################")
        """

    queries_dict_train, queries_dict_test, qrels_dict, qrels_dict_ssl, corpus_dict = get_formatted_data('mixed')

    qrels_combined = dict(qrels[0], **qrels[1])
    qrels_combined = dict(qrels_combined, **qrels[2])
    query_mixed = {}
    query_mixed_short = {}
    for id in queries_dict_test.keys():
        if id in qrels_dict:
            query_mixed[id] = queries_dict_test[id]
            if len(query_mixed) < 100:
                query_mixed_short[id] = queries_dict_test[id]
            if len(query_mixed) > 50000:
                break

    qrels_ssl_combined = dict(ssl_queries[0], **ssl_queries[1])
    qrels_ssl_combined = dict(qrels_ssl_combined, **ssl_queries[2])
    query_ssl = {}
    for id in queries_dict_test.keys():
        if id in qrels_ssl_combined:
            query_ssl[id] = queries_dict_test[id]
            if len(query_ssl) > 50000:
                break

    process_data = ProcessData()
    for language in ['german', 'french', 'italian']:
        query_s, corpus_s = process_data.remove_stopwords(query_mixed, corpus_dict, language)

    queries_dict_train, queries_dict_test, qrels_dict, qrels_dict_ssl, corpus_dict = get_formatted_data('mixed')

    corpus_short = {}
    qrel_short = {}
    for id in query_mixed_short.keys():
        qrel_short[id] = qrels_dict[id]
        for id in qrels_dict[id].keys():
            corpus_short[id] = corpus_dict[id]
    print(f"Length of shortened corpus: {len(corpus_short)}")
    print(f"Length of shortened qrel: {len(qrel_short)}")

    """
    # evaluate S-BERT for trained on hardneg: all, SSl, S, 100
    print("####################################################################################")
    print('mixed')
    model_name = 'Stern5497/sbert-distiluse-hardneg'
    print(model_name)
    print("####################################################################################")
    print('SBERT')
    run_evaluate_sbert(corpus_dict, query_mixed, qrels_dict, model_name)
    print("####################################################################################")
    print('SSL')
    run_evaluate_sbert(corpus_dict, query_ssl, qrels_ssl_combined, model_name)
    print("####################################################################################")
    print('S')
    run_evaluate_sbert(corpus_s, query_s, qrels_combined, model_name)
    print("####################################################################################")
    print('100')
    run_evaluate_sbert(corpus_short, query_mixed_short, qrel_short, model_name)
    print("####################################################################################")

    print("####################################################################################")
    print('mixed')
    model_name = 'Stern5497/sBert-swiss-legal-base'
    print(model_name)
    print("####################################################################################")
    print('SBERT')
    run_evaluate_sbert(corpus_dict, query_mixed, qrels_dict, model_name)
    print("####################################################################################")
    print('SSL')
    run_evaluate_sbert(corpus_dict, query_ssl, qrels_ssl_combined, model_name)
    print("####################################################################################")
    print('S')
    run_evaluate_sbert(corpus_s, query_s, qrels_combined, model_name)
    print("####################################################################################")

    model_name = 'distiluse-base-multilingual-cased-v1'  # multilingual model - not trained
    print(model_name)
    print("####################################################################################")
    print('dim reduction')
    run_evaluate_dim_reduction(corpus_dict, query_mixed, qrels_dict, model_name)
    print("####################################################################################")
    print('SBERT')
    run_evaluate_sbert(corpus_dict, query_mixed, qrels_dict, model_name)
    print("####################################################################################")
    print('SSL')
    run_evaluate_sbert(corpus_dict, query_ssl, qrels_ssl_combined, model_name)
    print("####################################################################################")
    print('S')
    run_evaluate_sbert(corpus_s, query_s, qrels_combined, model_name)
    print("####################################################################################")

    print("####################################################################################")
    model_name = 'Stern5497/sbert-legal-swiss-roberta-base'
    print(model_name)
    print("####################################################################################")
    print('SBERT')
    run_evaluate_sbert(corpus_dict, query_mixed, qrels_dict, model_name)
    print("####################################################################################")
    print('SSL')
    run_evaluate_sbert(corpus_dict, query_ssl, qrels_ssl_combined, model_name)
    print("####################################################################################")
    print('S')
    run_evaluate_sbert(corpus_s, query_s, qrels_combined, model_name)
    print("####################################################################################")

    print("####################################################################################")
    model_name = 'Stern5497/sBert-legal-xlm-base'
    print(model_name)
    print("####################################################################################")
    print('SBERT')
    run_evaluate_sbert(corpus_dict, query_mixed, qrels_dict, model_name)
    print("####################################################################################")
    print('SSL')
    run_evaluate_sbert(corpus_dict, query_ssl, qrels_ssl_combined, model_name)
    print("####################################################################################")
    print('S')
    run_evaluate_sbert(corpus_s, query_s, qrels_combined, model_name)
    print("####################################################################################")

    print("####################################################################################")
    model_name = 'Stern5497/sbert-legal-xlm-roberta-base'
    print(model_name)
    print("####################################################################################")
    print('SBERT')
    run_evaluate_sbert(corpus_dict, query_mixed, qrels_dict, model_name)
    print("####################################################################################")
    print('SSL')
    run_evaluate_sbert(corpus_dict, query_ssl, qrels_ssl_combined, model_name)
    print("####################################################################################")
    print('S')
    run_evaluate_sbert(corpus_s, query_s, qrels_combined, model_name)
    print("####################################################################################")

    model_name = 'Stern5497/sbert-distiluse'  # multilingual model - not trained
    print(model_name)
    print('SBERT')
    run_evaluate_sbert(corpus_dict, query_mixed, qrels_dict, model_name)
    print("####################################################################################")
    print('SSL')
    run_evaluate_sbert(corpus_dict, query_ssl, qrels_ssl_combined, model_name)
    print("####################################################################################")
    print('S')
    run_evaluate_sbert(corpus_s, query_s, qrels_combined, model_name)
    print("####################################################################################")
    print("####################################################################################")
    print('dim reduction')
    run_evaluate_dim_reduction(corpus_dict, query_mixed, qrels_dict, model_name)
    print("####################################################################################")


    print("####################################################################################")
    model_name = 'Stern5497/sbert-distiluse'  # multilingual model - not trained
    print('dim reduction')
    run_evaluate_dim_reduction(corpus_short, query_mixed_short, qrel_short, model_name)

    print("####################################################################################")
    model_name = 'Stern5497/sbert-distiluse'  # multilingual model - not trained
    print(model_name)
    print('SBERT')
    run_evaluate_sbert(corpus_short, query_mixed_short, qrel_short, model_name)
    print("####################################################################################")
    print("####################################################################################")
    model_name = 'Stern5497/sbert-legal-xlm-roberta-base'
    print(model_name)
    print("####################################################################################")
    print('SBERT')
    run_evaluate_sbert(corpus_short, query_mixed_short, qrel_short, model_name)
    print("####################################################################################")
    print("####################################################################################")
    model_name = 'Stern5497/sbert-legal-swiss-roberta-base'
    print(model_name)
    print("####################################################################################")
    print('SBERT')
    run_evaluate_sbert(corpus_short, query_mixed_short, qrel_short, model_name)
    print("####################################################################################")
    print("####################################################################################")
    model_name = 'distiluse-base-multilingual-cased-v1'  # multilingual model - not trained
    print(model_name)
    print("####################################################################################")
    print('dim reduction')
    run_evaluate_dim_reduction(corpus_short, query_mixed_short, qrel_short, model_name)
    print("####################################################################################")
    print("####################################################################################")
    print('SBERT')
    run_evaluate_sbert(corpus_short, query_mixed_short, qrel_short, model_name)
    print("####################################################################################")
    """

    print("####################################################################################")
    print('mixed')
    model_name = 'Stern5497/sbert-distiluse-hardneg'
    print(model_name)
    print('SB 100 only')
    run_evaluate_sbert(corpus_short, query_mixed_short, qrel_short, model_name)
    print("####################################################################################")
    model_name = 'distiluse-base-multilingual-cased-v1'  # multilingual model - not trained
    print(model_name)
    print('dim reduction 100')
    run_evaluate_dim_reduction(corpus_short, query_mixed_short, qrel_short, model_name)
    print("####################################################################################")
    print('SB 100 only')
    run_evaluate_sbert(corpus_short, query_mixed_short, qrel_short, model_name)
    print("####################################################################################")
    model_name = 'Stern5497/sbert-distiluse'  # multilingual model - not trained
    print(model_name)
    print('SB 100 only')
    run_evaluate_sbert(corpus_short, query_mixed_short, qrel_short, model_name)
    print("####################################################################################")
    model_name = 'Stern5497/sbert-legal-swiss-roberta-base'
    print(model_name)
    print('SB 100 only')
    run_evaluate_sbert(corpus_short, query_mixed_short, qrel_short, model_name)
    print("####################################################################################")
    model_name = 'Stern5497/sbert-legal-xlm-roberta-base'
    print(model_name)
    print('SBERT')
    run_evaluate_sbert(corpus_short, query_mixed_short, qrel_short, model_name)


if __name__ == '__main__':
    print("Start")
    # run_project()
    """
    model_name = 'Stern5497/sbert-distiluse'
    queries_dict_train, queries_dict_test, qrels_dict, qrels_dict_ssl, corpus_dict = get_formatted_data('mixed')
    run_train_sbert_hardneg(corpus_dict, queries_dict_train, qrels_dict, model_name, from_pretrained=True)
    """
    model_name = 'joelito/legal-swiss-roberta-base'
    queries_dict_train, queries_dict_test, qrels_dict, qrels_dict_ssl, corpus_dict = get_formatted_data('mixed')
    run_train_sbert_hardneg(corpus_dict, queries_dict_train, qrels_dict, model_name, from_pretrained=True)

    """
    print("####################################################################################")
    queries_dict_train, queries_dict_test, qrels_dict, qrels_dict_ssl, corpus_dict = get_formatted_data('mixed')
    train_models(corpus_dict, queries_dict_train, qrels_dict)
    print("####################################################################################")
    
    queries_dict_train, queries_dict_test, qrels_dict, qrels_dict_ssl, corpus_dict = get_formatted_data('mixed')
    model_name = 'distiluse-base-multilingual-cased-v1'
    train_loss = 'cosine'  # cosine or dot_product
    train(corpus_dict, queries_dict_train, qrels_dict, None, None, None, model_name=model_name, train_loss=train_loss,
          pretrained_model=True)
    """


