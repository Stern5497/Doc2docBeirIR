'''
This examples show how to train a basic Bi-Encoder for any BEIR dataset without any mined hard negatives or triplets.
The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.
For training, we use MultipleNegativesRankingLoss. There, we pass pairs in the format:
(query, positive_passage). Other positive passages within a single batch becomes negatives given the pos passage.
We do not mine hard negatives or train triplets in this example.
Running this script:
python train_sbert.py
'''

from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os
import logging
from time import time
import random
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models as md


#### Just some code to print debug information to stdout
from process_data import ProcessData

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download nfcorpus.zip dataset and unzip the dataset
dataset = "nfcorpus"

process_data = ProcessData()
corpus = process_data.load_corpus("data/corpus.jsonl")
qrels = process_data.load_qrels("data/qrels.jsonl")
train_queries, test_queries, val_queries = process_data.load_queries("data/queries.jsonl")

#### Please Note not all datasets contain a dev split, comment out the line if such the case
# dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")

#### Provide any sentence-transformers or HF model
#model_name = "bert-base-uncased"
#word_embedding_model = models.Transformer(model_name, max_seq_length=350)
#pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
#model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

#### Or provide pretrained sentence-transformer model
model = SentenceTransformer("distiluse-base-multilingual-cased")

retriever = TrainRetriever(model=model, batch_size=16)

#### Prepare training samples
train_samples = retriever.load_train(corpus, train_queries, qrels)
train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

#### Training SBERT with cosine-product
train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
#### training SBERT with dot-product
# train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.dot_score)

#### Prepare dev evaluator
# ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)

#### If no dev set is present from above use dummy evaluator
ir_evaluator = retriever.load_dummy_evaluator()

#### Provide model save path
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "{}-v1-{}".format(model_name, dataset))
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


###### EVALUATE MODEL ##################
"""model = DRES(md.SentenceBERT("msmarco-distilbert-base-tas-b"), batch_size=256, corpus_chunk_size=512*9999)

retriever = EvaluateRetrieval(model, score_function="dot")

#### Retrieve dense results (format of results is identical to qrels)
start_time = time()
results = retriever.retrieve(corpus, test_queries)
end_time = time()
print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
#### Evaluate your retrieval using NDCG@k, MAP@K ...

logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="r_cap")
hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")

#### Print top-k documents retrieved ####
top_k = 10

query_id, ranking_scores = random.choice(list(results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info("Query : %s\n" % test_queries[query_id])

for rank in range(top_k):
    doc_id = scores_sorted[rank][0]
    # Format: Rank x: ID [Title] Body
    logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))


"""

