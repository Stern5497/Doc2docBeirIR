import wandb
import json
import os, pathlib
import beir
from beir.datasets.data_loader import GenericDataLoader
from train_model import train
from beir import util, LoggingHandler
from PreprocessDoc2doc import PreprocessDoc2doc


"""
We are using https://github.com/beir-cellar/beir/wiki/Examples-and-tutorials as an example and work with their built in 
ir models



See this colab as an example how to use ir models of beir
https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing#scrollTo=B2X_deOMFxJB

"""


def run_project():
    preprocessDoc2doc = PreprocessDoc2doc()
    data_dict = preprocessDoc2doc.create_data()

    corpus = data_dict['corpus']
    queries = data_dict['queries']['validation']
    qrels = data_dict['qrels']['validation']
    train(corpus, queries, qrels)


if __name__ == '__main__':
    run_project()
