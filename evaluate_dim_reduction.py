"""
The pre-trained models produce embeddings of size 512 - 1024. However, when storing a large
number of embeddings, this requires quite a lot of memory / storage.
In this example, we reduce the dimensionality of the embeddings to e.g. 128 dimensions. This significantly
reduces the required memory / storage while maintaining nearly the same performance.
For dimensionality reduction, we compute embeddings for a large set of (representative) sentence. Then,
we use PCA to find e.g. 128 principle components of our vector space. This allows us to maintain
us much information as possible with only 128 dimensions.
PCA gives us a matrix that down-projects vectors to 128 dimensions. We use this matrix
and extend our original SentenceTransformer model with this linear downproject. Hence,
the new SentenceTransformer model will produce directly embeddings with 128 dimensions
without further changes needed.
Usage: python evaluate_dim_reduction.py
"""

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import PCAFaissSearch

import logging
import pathlib, os
import random
import faiss

def run_evaluate_dim_reduction(corpus, queries, qrels, model_name):

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    dataset = "doc2doc"

    # Dense Retrieval using Different Faiss Indexes (Flat or ANN) ####
    # Provide any Sentence-Transformer or Dense Retriever model.

    # model_path = "msmarco-distilbert-base-tas-b"

    # Use a multilingual sentence-transformer:
    # model_path = 'sentence-transformers/distiluse-base-multilingual-cased-v1'
    # model_path = "paraphrase-albert-small-v2"
    model = models.SentenceBERT(model_name)

    ###############################################################
    #### PCA: Principal Component Analysis (Exhaustive Search) ####
    ###############################################################
    # Reduce Input Dimension (768) to output dimension of (128)

    output_dimension = 128
    base_index = faiss.IndexFlatIP(output_dimension)
    faiss_search = PCAFaissSearch(model,
                                  base_index=base_index,
                                  output_dimension=output_dimension,
                                  batch_size=128)

    #######################################################################
    #### PCA: Principal Component Analysis (with Product Quantization) ####
    #######################################################################
    # Reduce Input Dimension (768) to output dimension of (96)

    # output_dimension = 96
    # base_index = faiss.IndexPQ(output_dimension,               # output dimension
    #                              96,                           # number of centroids
    #                              8,                            # code size
    #                              faiss.METRIC_INNER_PRODUCT)   # similarity function

    # faiss_search = PCAFaissSearch(model,
    #                               base_index=base_index,
    #                               output_dimension=output_dimension,
    #                               batch_size=128)

    #### Load faiss index from file or disk ####
    # We need two files to be present within the input_dir!
    # 1. input_dir/{prefix}.{ext}.faiss => which loads the faiss index.
    # 2. input_dir/{prefix}.{ext}.faiss => which loads mapping of ids i.e. (beir-doc-id \t faiss-doc-id).

    prefix = "my-index"  # (default value)
    ext = "pca"  # extension

    input_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "faiss-index")

    if os.path.exists(os.path.join(input_dir, "{}.{}.faiss".format(prefix, ext))):
        faiss_search.load(input_dir=input_dir, prefix=prefix, ext=ext)

    #### Retrieve dense results (format of results is identical to qrels)
    retriever = EvaluateRetrieval(faiss_search, score_function="dot")  # or "cos_sim"
    results = retriever.retrieve(corpus, queries)

    ### Save faiss index into file or disk ####
    # Unfortunately faiss only supports integer doc-ids, We need save two files in output_dir.
    # 1. output_dir/{prefix}.{ext}.faiss => which saves the faiss index.
    # 2. output_dir/{prefix}.{ext}.faiss => which saves mapping of ids i.e. (beir-doc-id \t faiss-doc-id).

    prefix = "my-index"  # (default value)
    ext = "pca"  # extension

    output_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "faiss-index")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(os.path.join(output_dir, "{}.{}.faiss".format(prefix, ext))):
        faiss_search.save(output_dir=output_dir, prefix=prefix, ext=ext)

    #### Evaluate your retrieval using NDCG@k, MAP@K ...

    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
    recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="r_cap")
    hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")
