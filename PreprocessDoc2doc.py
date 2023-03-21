import datasets
import json
import gc
import pandas as pd
import ast
from datasets import load_dataset
from tqdm import tqdm


class PreprocessDoc2doc:
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
        self.feature_cols = ['facts', 'considerations']
        self.id_list = []

    def create_data(self):
        data = {}
        data['corpus'] = self.create_corpus()
        dataset_dict = self.load_dataset()
        collected = gc.collect()
        print("Garbage collector: collected %d objects." % collected)
        for key, value in dataset_dict.items():
            print(f"Processing {key}")
            if 'queries' not in data:
                data['queries'] = {key: self.create_queries(value)}
                data['qrels'] = {key: self.create_qrels(value)}
            else:
                data['queries'][key] = self.create_queries(value)
                data['qrels'][key] = self.create_qrels(value)
        collected = gc.collect()
        print("Garbage collector: collected %d objects." % collected)
        return data

    def load_dataset(self):
        # load laws from huggingface
        dataset_dict = dict()
        dataset_dict['train'] = load_dataset('rcds/doc2doc', split='train')
        dataset_dict['validation'] = load_dataset('rcds/doc2doc', split='validation')
        dataset_dict['test'] = load_dataset('rcds/doc2doc', split='test')
        print(dataset_dict['train'])
        print(dataset_dict['validation'])
        print(dataset_dict['test'])

        for k, v in dataset_dict.items():
            columns_to_remove = [item for item in dataset_dict[k].column_names if
                                 item not in ['decision_id', 'facts', 'considerations', 'laws', 'cited_rulings']]
            dataset_dict[k] = dataset_dict[k].remove_columns(columns_to_remove)
            df = pd.DataFrame(dataset_dict[k])
            # TODO remove this
            self.id_list = ['bd79de92-c383-428b-80ba-3aa520bc314b', '7c774696-2a73-4615-95fe-435ed872159d',
             'cd7b72a1-2345-49b9-826c-75b821e3c2b9','86530c8d-c9cf-44a2-b39d-03c825376ff1', 'b78c4d4f-2ac2-485e-a02b-011cfd250dc8',
             '056a3cc4-162d-439c-bd55-40b6bc33d61c', 'f3c94596-29a4-491c-a90f-a0600c0b76d9', '1ed2737f-f130-4dc7-aa4c-b5393ff72a7f',
             'aae2e585-61f5-4635-a486-ecdd038e8f13']
            dataset_dict[k] = self.clean_dataset(df)
        return dataset_dict

    def clean_dataset(self, df):
        df['cited_rulings'] = df.cited_rulings.apply(lambda x: [item for item in ast.literal_eval(x) if item in self.id_list])
        df['laws'] = df.laws.apply(lambda x: [item for item in ast.literal_eval(x) if item in self.id_list])
        combinate = lambda s1, s2: list(s1) + list(s2)
        df['citations'] = df['laws'].combine(df['cited_rulings'], combinate)
        df['citations'] = df.citations.apply(lambda x: None if x == [] else x)
        df.drop(columns=['cited_rulings', 'laws'], inplace=True)
        df = df.dropna()
        return df

    def create_corpus(self):
        # process laws
        law_dataset = load_dataset('rcds/swiss_legislation', split='train')
        law_dataset = law_dataset.filter(lambda row: row['canton'] == 'ch')
        cols_to_remove = [item for item in law_dataset.column_names if item not in ['uuid', 'sr_number', 'pdf_content']]
        law_dataset = law_dataset.remove_columns(cols_to_remove)
        law_dataset = law_dataset.rename_column("uuid", "id")
        law_dataset = law_dataset.rename_column("sr_number", "title")
        law_dataset = law_dataset.rename_column("pdf_content", "text")

        law_df = pd.DataFrame(law_dataset)
        law_df['id'] = law_df.id.astype(str)
        self.id_list.extend(law_df.id.tolist())

        corpus_dict = dict()

        def write_dict(row):
            id = str(row['id'])
            corpus_dict[id] = {"title": row['title'], "text": row['text']}
            return row

        law_df.apply(write_dict, axis="columns")

        # process rulings
        rulings_dict = self.load_rulings()
        self.id_list.extend(list(rulings_dict.keys()))

        corpus_dict = {**corpus_dict, **rulings_dict}

        return corpus_dict

    def load_rulings(self):
        rulings_dict = {}
        num_lines = sum(1 for i in open('data/rulings.jsonl', 'rb'))
        with open('data/rulings.jsonl', encoding='utf8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                id = line.get("decision_id")
                file = line.get("file_number")
                facts = line.get("facts")
                considerations = line.get("considerations")
                rulings_dict[id] = {"title": file, "text": f"{facts} {considerations}"}
        return rulings_dict

    def create_queries(self, df):
        """
        queries = {
            "decision_id_1": "facts or considerations",
            "decision_id_2": "facts or considerations"
        }
        """
        queries = {}
        for feature in ['facts', 'considerations']:
            print(f"Prozessing {feature}")
            queries_dict = {}

            def write_queries_dict(row):
                id = row['decision_id']
                queries_dict[id] = str(row[feature])
                return row

            df = df.apply(write_queries_dict, axis="columns")
            print("Successfully created queries dict.")
            queries[feature] = queries_dict
            # TODO change handle both features
        return queries_dict


    def create_qrels(self, df):
        """
        qrels = {
            "decision_id_1": {"law_id_1": 1},
            "decision_id_2": {"decision_id_bge_1F": 1},
        }
        """
        df.drop(columns=['facts', 'considerations'], inplace=True)
        df = df.explode('citations')
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


if __name__ == '__main__':
    preprocessDoc2doc = PreprocessDoc2doc()
    preprocessDoc2doc.create_data()
