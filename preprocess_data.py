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


class PreprocessData:
    """
    corpus = {
        "law_id_1": {
            "title": "sr_number",
            "text": "blabliblu“
        },
        "decision_id_bge_1": {
            "title": "sr_number“,
            "text": "blablebli“
    },
    }

    queries = {
        "decision_id_1": "facts or considerations",
        "decision_id_2": "facts or considerations"
    }

    qrels = {
        "decision_id_1": {"law_id_1": 1},
        "decision_id_2": {"decision_id_bge_1": 1},
    }
    important all ids used in qrels must be found in corpus
    """

    def __init__(self):
        self.counter = 0

    def get_dataset(self):
        # load laws from huggingface
        dataset_train = load_dataset('rcds/doc2doc', split='train')
        dataset_validation = load_dataset('rcds/doc2doc', split='validation')
        dataset_test = load_dataset('rcds/doc2doc', split='test')
        # dataset = interleave_datasets([dataset_train, dataset_validation, dataset_test])
        dataset = dataset_test.shuffle(seed=42)

        dataset = dataset.filter(lambda example: not example["cited_rulings"].startswith("[]"))

        # TODO only for debug:
        dataset = dataset.select(list(range(0, 100)))
        # TODO try only rulings or law to find out where error in finding ids occurs
        columns_to_remove = [item for item in dataset.column_names if
                             item not in ['decision_id', 'facts', 'considerations', 'laws', 'cited_rulings']]
        dataset = dataset.remove_columns(columns_to_remove)
        df = pd.DataFrame(dataset)
        df = self.clean_dataset(df)
        return df

    def clean_dataset(self, df):
        def create_list_citations(row):
            id_list = ast.literal_eval(row['cited_rulings']) + ast.literal_eval(row['laws'])
            id_list = list(set(id_list))
            return id_list

        df['citations'] = df.apply(create_list_citations, axis='columns')
        df['citations'] = df.citations.apply(lambda x: None if x == [] else x)
        df.drop(columns=['cited_rulings', 'laws'], inplace=True)
        df = df.dropna()
        return df

    def create_corpus(self):
        # process laws
        law_dataset = load_dataset('rcds/swiss_legislation', split='train')
        law_dataset = law_dataset.filter(lambda row: row['canton'] == 'ch')
        cols_to_remove = [item for item in law_dataset.column_names if
                          item not in ['uuid', 'sr_number', 'pdf_content']]
        law_dataset = law_dataset.remove_columns(cols_to_remove)
        law_dataset = law_dataset.rename_column("uuid", "id")
        law_dataset = law_dataset.rename_column("sr_number", "title")
        law_dataset = law_dataset.rename_column("pdf_content", "text")

        law_df = pd.DataFrame(law_dataset)
        law_df['id'] = law_df.id.astype(str)

        corpus_dict = dict()

        def write_dict(row):
            id = str(row['id'])
            corpus_dict[id] = {"title": row['title'], "text": row['text']}
            return row

        law_df.apply(write_dict, axis="columns")

        # process rulings
        rulings_df = self.load_rulings()

        def write_rulings_dict(row):
            id = str(row['decision_id'])
            corpus_dict[id] = {"title": row['file_number'], "text": f"{row['facts']} {row['considerations']}"}

        rulings_df = rulings_df.apply(write_rulings_dict, axis='columns')

        return corpus_dict

    def load_rulings(self):
        ruling_dataset = load_dataset('rcds/swiss_rulings', split='train')
        decision_df = pd.DataFrame(ruling_dataset)
        print(f"BGE: There are {len(decision_df.index)} in db (also old or not referenced included).")
        return decision_df

    def create_queries(self, df, feature):

        print(f"Prozessing {feature}")
        queries_dict = {}

        def write_queries_dict(row):
            id = row['decision_id']
            queries_dict[id] = str(row[feature])
            return row

        df = df.apply(write_queries_dict, axis="columns")
        print("Successfully created queries dict.")
        return queries_dict

    def create_qrels(self, df, corpus):

        df.drop(columns=['facts', 'considerations'], inplace=True)
        df = df.explode('citations')
        df = df.dropna()

        def filter_cits(cit):
            try:
                match = corpus[cit]
                return True
            except Exception as e:
                pass
            return None

        df['match'] = df.citations.apply(filter_cits)

        df = df.dropna()

        qrels_dict = {}

        def write_qrels_dict(row):
            query_id = row['decision_id']
            corpus_id = row['citations']
            if query_id not in qrels_dict:
                qrels_dict[query_id] = {corpus_id: 1}
            else:
                qrels_dict[query_id][corpus_id] = 1
            return row

        df = df.apply(write_qrels_dict, axis="columns")

        return qrels_dict

    def create_triplets(self, df, feature, corpus):
        """
        triplets:
        1. bger_text (facts or considerations)
        2. cited ruling or law text
        3. ruling or law which was NOT cited text
        """

        # create sample triplets:
        # 1. pick random 100 bger samples -> already done by providing df
        df['cit_list'] = df['citations']

        # 2. go explode bger for each cited law and ruling
        df = df.explode('citations')
        print(df.citations.tolist())

        # 3. find for each law / ruling the text

        def write_text(row):
            try:
                text = corpus[row['citations']]
                return text
            except Exception as e:
                pass
            return None

        df['citations'] = df.apply(write_text, axis='columns')

        # 4. find a good example of law / ruling not in citaions of that case
        def find_neg_example(row):
            not_found = True
            while not_found:
                corp_id, corp_text = random.choice(list(corpus.items()))
                if corp_id not in row['cit_list']:
                    not_found = False
            return corp_text['text']

        df['neg_text'] = df.apply(find_neg_example, axis='columns')

        # 5. get the right format  List[Tuple[str, str, str]]
        columns_to_remove = [item for item in df.columns if item not in [feature, 'citations', 'neg_text']]
        df = df.drop(columns=columns_to_remove)
        df = df.dropna()

        triples = []

        def write_list(row):
            triples.append((row[feature], row['citations'], row['neg_text']))

        df = df.apply(write_list, axis='columns')
        return triples

    def create_data(self):
        print("load corpus")
        data = {'corpus': self.create_corpus()}
        print("get data")
        df = self.get_dataset()
        data['queries'] = {'facts': self.create_queries(df, 'facts')}
        data['queries']['considerations'] = self.create_queries(df, 'considerations')
        data['qrels'] = self.create_qrels(df.copy(), data['corpus'])
        data['triples'] = {'facts': self.create_triplets(df, 'facts', data['corpus'])}
        data['triplets']['considerations'] = self.create_triplets(df, 'considerations', data['corpus'])