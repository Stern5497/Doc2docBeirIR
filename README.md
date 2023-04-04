Models to use:

- bm25: from beir.retrieval.search.lexical import BM25Search as BM25

  - dense: 
  from beir.retrieval import models 
  from beir.retrieval.evaluation import EvaluateRetrieval
  from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
  model = DRES(models.SentenceBERT("msmarco-distilbert-base-v3"), batch_size=16)