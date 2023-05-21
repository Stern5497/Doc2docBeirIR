'''
https://github.com/beir-cellar/beir/blob/main/examples/retrieval/training/train_sbert.py

This examples show how to train a basic Bi-Encoder for any BEIR dataset without any mined hard negatives or triplets.
The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.
For training, we use MultipleNegativesRankingLoss. There, we pass pairs in the format:
(query, positive_passage). Other positive passages within a single batch becomes negatives given the pos passage.
We do not mine hard negatives or train triplets in this example.
Running this script:
python train_sbert.py
'''
import pathlib, os
import logging
import wandb

from typing import Dict, Type, List, Callable, Iterable, Tuple
from tqdm.autonotebook import trange
from sentence_transformers.readers import InputExample

from sentence_transformers import losses, models, SentenceTransformer

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


def train(corpus, queries, qrels, dev_corpus, dev_queries, dev_qrels, model_name="xlm-roberta-base", train_loss='cosine', pretrained_model=True):

    dev_available = False
    if dev_corpus is not None and dev_queries is not None and dev_qrels is not None:
        dev_available = True

    dataset = 'doc2doc'

    #### Provide any sentence-transformers or HF model
    if not pretrained_model:
        word_embedding_model = models.Transformer(model_name, max_seq_length=350)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        print("successfully created model")
    
    #### Or provide pretrained sentence-transformer model
    else:
        model = SentenceTransformer(model_name)

    retriever = TrainRetriever(model=model, batch_size=16)
    print("successfully created retriever")

    """
        word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        
        
        word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())
    
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
    """

    #### Prepare training samples
    train_samples = retriever.load_train(corpus, queries, qrels)
    train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

    #### Training SBERT with cosine-product
    if train_loss == 'cosine':
        train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
    #### training SBERT with dot-product
    else:
        train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.dot_score)

    #### Prepare dev evaluator
    if dev_available:
        ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)
    #### If no dev set is present from above use dummy evaluator
    else:
        ir_evaluator = retriever.load_dummy_evaluator()

    #### Provide model save path
    model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output",
                                   "{}-v1-{}".format(model_name, dataset))
    os.makedirs(model_save_path, exist_ok=True)
    print("successfully created set up training")

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

    print("successfully did it")


