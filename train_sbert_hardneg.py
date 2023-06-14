'''
This examples show how to train a Bi-Encoder for any BEIR dataset.

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)

Negative passage are hard negative examples, that where retrieved by lexical search. We use Elasticsearch
to get (max=10) hard negative examples given a positive passage.

Running this script:
python train_sbert_BM25_hardnegs.py
'''
from time import time

from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.train import TrainRetriever
import pathlib, os, tqdm
import logging
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import models as smodels

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

def run_train_sbert_hardneg(corpus, queries, qrels, model_name, from_pretrained=True):

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    dataset = "doc2doc"

    # get cached checkpoint
    if from_pretrained:
       model = DRES(models.SentenceBERT((
                model_name,
                model_name,
                " [SEP] "), batch_size=128))

    else:
        model = DRES(models.SentenceBERT(model_name), batch_size=256, corpus_chunk_size=512*9999)

    retriever = EvaluateRetrieval(model, score_function="dot")
    results = retriever.retrieve(corpus, queries)
    print("successfully retreived results using SBERT")
    triplets = []
    hard_negatives_max = 5

    for query_id, query_text in queries.items():
        pos_ids = qrels[query_id].keys()  # positive corpus ids
        sbert_results = results[query_id]
        # go throuh all positive corpus ids and find negative ids
        for pos_id in pos_ids:
            found = 0
            for key, value in sbert_results.items():
                if (value < 30 and key not in pos_ids):
                    pos_text = corpus[pos_id]["text"]
                    neg_text = corpus[key]["text"]
                    triplets.append([query_text, pos_text, neg_text])
                    found = found + 1
                    if found > hard_negatives_max:
                        break


    #### Provide any sentence-transformers or HF model
    word_embedding_model = smodels.Transformer(model_name, max_seq_length=512)
    pooling_model = smodels.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # model = SentenceTransformer(model_name)

    #### Provide a high batch-size to train better with triplets!
    retriever = TrainRetriever(model=model, batch_size=16)

    #### Prepare triplets samples
    train_samples = retriever.load_train_triplets(triplets=triplets)
    train_dataloader = retriever.prepare_train_triplets(train_samples)

    #### Training SBERT with cosine-product
    train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)

    #### training SBERT with dot-product
    # train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.dot_score)

    #### Prepare dev evaluator
    # ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)

    #### If no dev set is present from above use dummy evaluator
    ir_evaluator = retriever.load_dummy_evaluator()

    #### Provide model save path
    model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "{}-v2-{}-bm25-hard-negs".format(model_name, dataset))
    os.makedirs(model_save_path, exist_ok=True)

    #### Configure Train params
    num_epochs = 1
    evaluation_steps = 5000
    warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

    retriever.fit(train_objectives=[(train_dataloader, train_loss)],
                    evaluator=ir_evaluator,
                    epochs=num_epochs,
                    output_path=model_save_path,
                    warmup_steps=warmup_steps,
                    evaluation_steps=evaluation_steps,
                    use_amp=True)