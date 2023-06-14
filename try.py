import wandb
import json
import os, pathlib
import ast
import random
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from transformers import AutoTokenizer



def create_df(dataset, attribute):
    dataset = dataset.filter(lambda row: row[attribute] is not None and row[attribute] != '')
    value_list = get_unique_values_dataset_column(dataset, attribute)
    data = []
    for value in value_list:
        amount = len(dataset.filter(lambda row: row[attribute] == value))
        data.append([value, amount])
    return pd.DataFrame(data, columns=[attribute, 'number of decisions'])


def get_unique_values_dataset_column(dataset, column_name):
    value_list = []

    def get_column_value(row):
        if row[column_name] not in value_list:
            value_list.append(row[column_name])

    dataset.map(get_column_value)
    return value_list

def get_rid_of_unused_cols(ds, col_list):
    cols = ds.column_names
    for item in cols:
        if item not in col_list:
            ds.remove_columns(item)
    return ds

def get_data(name, split, task=None):
    if task is None:
        dataset = load_dataset(name, split=split)
        print(dataset)
    else:
        datasets = load_dataset(name, task)
        if len(task)>10:
            datasets = datasets.filter(lambda example: len(example['input']) > 0)
            dataset = concatenate_datasets([datasets['train'], datasets['validation'], datasets['test']])
        print(dataset)

    return dataset

def run_project(name, split, columns, detail_filter=False, col='', compare='equal', value='', detail=[], task=None):
    print(name)
    dataset = get_data(name, split, task=task)
    cal_detail(dataset, columns, detail_filter, col, compare, value, detail)
    return dataset


def cal_detail(dataset, columns, detail_filter=False, col='', compare='equal', value='', detail=[]):
    for column in columns:
        df = create_df(dataset, column)
        print(df)
        pass
    if detail_filter:
        if compare == 'equal':
            filtered = dataset.filter(lambda example: example[col] == value)
        elif compare == 'bigger':
            filtered = dataset.filter(lambda example: example[col] >= value)
        print(filtered)
        for column in detail:
            df = create_df(filtered, column)
            print(df)

def cal_label(dataset, label, value_list):
    df = pd.DataFrame(dataset)
    data = []
    for value in value_list:
        match = df[label] == value
        amount = len(df[match].index)
        data.append([value, amount])
    print(data)
    return True


def count_cits(dataset, attributes):
    for attribute in attributes:
        def counter(example):
            num = len(list(ast.literal_eval(example[attribute])))
            example[attribute] = num
            return example
        dataset = dataset.map(counter)
        df = pd.DataFrame(dataset)
        leng = len(df.index)
        df = df[df[attribute] > 0]
        print(f"Found {leng-len(df.index)} decisions with no {attribute}")
        stats = df[attribute].describe()
        print(stats)


if __name__ == '__main__':
    print("Start")
    # run_project('rcds/swiss_legislation', 'train', ['canton', 'language'], True, 'canton', equal, 'ch', ['language])
    # run_project('rcds/swiss_rulings', 'train', ['language', 'chamber'], True, 'year', 'bigger', 2023, ['language'])
    # run_project('rcds/swiss_doc2doc_ir', 'train', ['language', 'chamber'])
    # run_project('rcds/swiss_doc2doc_ir', 'test', ['language', 'chamber'])
    # run_project('rcds/swiss_doc2doc_ir', 'validation', ['language', 'chamber'])

    """
    print('validation')
    dataset = get_data('rcds/swiss_doc2doc_ir', 'validation')
    count_cits(dataset, ['cited_rulings', 'laws'])
    print('test')
    dataset = get_data('rcds/swiss_doc2doc_ir', 'test')
    count_cits(dataset, ['cited_rulings', 'laws'])
    print('train')
    dataset = get_data('rcds/swiss_doc2doc_ir', 'train')
    count_cits(dataset, ['cited_rulings', 'laws'])
    """
    """for split in ['test', 'train', 'validation']:
        print('##########################################################################')
        print(split)
        print('swiss_criticality_prediction_bge_facts')
        dataset = load_dataset("joelito/lextreme", 'swiss_criticality_prediction_bge_facts', split=split)
        dataset = dataset.filter(lambda row: len(str(row['input'])) > 0)
        print(dataset)
        cal_label(dataset, 'label', [0, 1])
        print('swiss_criticality_prediction_bge_considerations')
        dataset = load_dataset("joelito/lextreme", 'swiss_criticality_prediction_bge_considerations', split=split)
        print(dataset)
        dataset = dataset.filter(lambda row: len(str(row['input'])) > 0)
        cal_label(dataset, 'label', [0, 1])
        print('swiss_criticality_prediction_citation_facts')
        dataset = load_dataset("joelito/lextreme", 'swiss_criticality_prediction_citation_facts', split=split)
        dataset = dataset.filter(lambda row: len(str(row['input'])) > 0)
        print(dataset)
        cal_label(dataset, 'label', [1,2,3])
        print('swiss_criticality_prediction_citation_considerations')
        dataset = load_dataset("joelito/lextreme", 'swiss_criticality_prediction_citation_considerations', split=split)
        dataset = dataset.filter(lambda row: len(str(row['input'])) > 0)
        cal_label(dataset, 'label', [1,2,3])
        print(dataset)

    
    run_project('rcds/swiss_criticality_prediction', 'train', ['language', 'year', 'law_area'])
    run_project('rcds/swiss_criticality_prediction', 'test', ['language', 'year', 'law_area'])
    run_project('rcds/swiss_criticality_prediction', 'validation', ['language', 'year', 'law_area'])
    """
    """
    run_project('rcds/swiss_criticality_prediction', 'train', [], True, col='language', compare='equal', value='de',
                detail=['bge_label', 'citation_label'])
    run_project('rcds/swiss_criticality_prediction', 'train', [], True, col='language', compare='equal', value='fr',
                detail=['bge_label', 'citation_label'])
    run_project('rcds/swiss_criticality_prediction', 'train', [], True, col='language', compare='equal', value='it',
                detail=['bge_label', 'citation_label'])
    run_project('rcds/swiss_criticality_prediction', 'test', [], True, col='language', compare='equal', value='de',
                detail=['bge_label', 'citation_label'])
    run_project('rcds/swiss_criticality_prediction', 'test', [], True, col='language', compare='equal', value='fr',
                detail=['bge_label', 'citation_label'])
    run_project('rcds/swiss_criticality_prediction', 'test', [], True, col='language', compare='equal', value='it',
                detail=['bge_label', 'citation_label'])
    run_project('rcds/swiss_criticality_prediction', 'validation', [], True, col='language', compare='equal',
                value='de', detail=['bge_label', 'citation_label'])
    run_project('rcds/swiss_criticality_prediction', 'validation', [], True, col='language', compare='equal',
                value='de', detail=['bge_label', 'citation_label'])
    run_project('rcds/swiss_criticality_prediction', 'validation', [], True, col='language', compare='equal',
                value='de', detail=['bge_label', 'citation_label'])
    

    run_project('rcds/swiss_judgment_prediction_xl', 'train', ['language', 'year', 'law_area', 'label', 'canton'])
    run_project('rcds/swiss_judgment_prediction_xl', 'test', ['language', 'year', 'law_area', 'label', 'canton'])
    run_project('rcds/swiss_judgment_prediction_xl', 'validation', ['language', 'year', 'law_area', 'label', 'canton'])

    run_project('rcds/swiss_law_area_prediction', 'train', ['language', 'year'])
    run_project('rcds/swiss_law_area_prediction', 'test', ['language', 'year'])
    run_project('rcds/swiss_law_area_prediction', 'validation', ['language', 'year'])
    

    run_project('joelito/lextreme', 'train', ['language', 'label' ], task='swiss_judgment_prediction_xl_facts')
    run_project('joelito/lextreme', 'test', ['language',  'label'], task='swiss_judgment_prediction_xl_facts')
    run_project('joelito/lextreme', 'validation', ['language', 'label'], task='swiss_judgment_prediction_xl_facts')

    run_project('joelito/lextreme', 'train', ['language', 'label'], task='swiss_judgment_prediction_xl_considerations')
    run_project('joelito/lextreme', 'test', ['language', 'label'], task='swiss_judgment_prediction_xl_considerations')
    run_project('joelito/lextreme', 'validation', ['language', 'label'], task='swiss_judgment_prediction_xl_considerations')

    run_project('joelito/lextreme', 'train', ['language', 'label'], task='swiss_law_area_prediction_facts')
    run_project('joelito/lextreme', 'test', ['language', 'label'], task='swiss_law_area_prediction_facts')
    run_project('joelito/lextreme', 'validation', ['language', 'label'],task='swiss_law_area_prediction_facts')

    run_project('joelito/lextreme', 'train', ['language', 'label'], task='swiss_law_area_prediction_considerations')
    run_project('joelito/lextreme', 'test', ['language', 'label'], task='swiss_law_area_prediction_considerations')
    run_project('joelito/lextreme', 'validation', ['language', 'label'], task='swiss_law_area_prediction_considerations')
    

    run_project('rcds/swiss_court_view_generation', 'train', ['language'], task='full')
    run_project('rcds/swiss_court_view_generation', 'test', ['language'], task='full')
    run_project('rcds/swiss_court_view_generation', 'validation', ['language'], task='full')

    run_project('rcds/swiss_court_view_generation', 'train', ['language'], task='origin')
    run_project('rcds/swiss_court_view_generation', 'test', ['language'], task='origin')
    run_project('rcds/swiss_court_view_generation', 'validation', ['language'], task='origin')
    
    run_project('joelito/lextreme', 'train', ['language', 'label'], task='swiss_law_area_prediction_facts')

    run_project('joelito/lextreme', 'train', ['language', 'label'], task='swiss_law_area_prediction_considerations')

    run_project('joelito/lextreme', 'train', ['language', 'label'], task='swiss_law_area_prediction_sub_area_facts')

    run_project('joelito/lextreme', 'train', ['language', 'label'], task='swiss_law_area_prediction_sub_area_considerations')

    run_project('joelito/lextreme', 'train', ['language', 'label'],
                task='swiss_law_area_prediction_public_considerations')
    run_project('joelito/lextreme', 'test', ['language', 'label'],
                task='swiss_law_area_prediction_public_considerations')
    run_project('joelito/lextreme', 'validation', ['language', 'label'],
                task='swiss_law_area_prediction_public_considerations')

    run_project('rcds/swiss_court_view_generation', 'train', ['language'], task='origin')
    run_project('rcds/swiss_court_view_generation', 'test', ['language'], task='origin')
    run_project('rcds/swiss_court_view_generation', 'validation', ['language'], task='origin')
    

    dataset = run_project('rcds/swiss_leading_decisions', 'train', ['language'])
    df = pd.DataFrame(dataset)
    print('facts')
    print(df.loc[:, 'facts_num_tokens_bert'].mean())
    print('considerations')
    print(df.loc[:, 'considerations_num_tokens_bert'].mean())

    dataset = run_project('rcds/swiss_rulings', 'train', ['language'])
    

    querie_dataset = load_dataset("Stern5497/querie")
    qrel_dataset = load_dataset("Stern5497/qrel")
    corpus_dataset = load_dataset("Stern5497/corpus")

    querie_dataset = querie_dataset['train']
    qrel_dataset = qrel_dataset['train']
    corpus_dataset = corpus_dataset['train']

    print(querie_dataset)
    print(qrel_dataset)
    print(corpus_dataset)
   
   
   #################################################################################################################
    """
    """
    
    
    def get_dataset_per_lang(name, splits, config):
        if config != '':
            multiple_ds = load_dataset(name, config)
        else:
            multiple_ds = load_dataset(name)
        if len(splits)>1:
            i=1
            single_ds = multiple_ds[splits[0]]
            while i < len(splits):
                single_ds = concatenate_datasets([single_ds, multiple_ds[splits[i]]])
                i = i+1
        else:
            single_ds = multiple_ds[splits[0]]
        if name == 'rcds/swiss_legislation':
            single_ds = single_ds.filter(lambda row: row['canton'] == 'ch')
        ds = {}
        ds['de'] = single_ds.filter(lambda row: row['language'] == 'de')
        print(f"de: {ds['de']}")
        ds['fr'] = single_ds.filter(lambda row: row['language'] == 'fr')
        print(f"fr: {ds['fr']}")
        ds['it'] = single_ds.filter(lambda row: row['language'] == 'it')
        print(f"it: {ds['it']}")
        print("successfully got datasets")
        return ds

    def get_tokenizers(lang):
        os.environ['TOKENIZERS_PARALLELISM'] = "True"
        if lang == 'de':
            bert = "deepset/gbert-base"
        elif lang == 'fr':
            bert = "camembert/camembert-base-ccnet"
        elif lang == 'it':
            bert = "dbmdz/bert-base-italian-cased"
        else:
            raise ValueError(
                f"Please choose another language.")
        return AutoTokenizer.from_pretrained(bert)

    def count_tokens(lang, dataset, column):
        bert_tokenizer=get_tokenizers(lang)
        print("got tokenizer")
        df = pd.DataFrame(dataset)
        df[column] = df[column].tolist()
        def get_it(row):
            text = row[column]
            ids = bert_tokenizer(text).input_ids
            length = len(ids)
            return length

        df['num_tokens_bert'] = df.apply(get_it, axis='columns')
        return df

    def token_length_per_dataset(name='rcds/swiss_legislation', splits=['train'], columns=['pdf_content'], config=''):
        print(f"Prozessing dataset {name} with configuration {config}")
        ds = get_dataset_per_lang(name=name, splits=splits, config=config)
        for column in columns:
            print(f"Prozessing column: {column}")
            df = pd.DataFrame()
            for key, value in ds.items():
                print("start for df")
                tmp_df = count_tokens(key, value, column)
                df = pd.concat([df, tmp_df])
            print("Mean token length:")
            print(df.loc[:, 'num_tokens_bert'].mean())
            print("95% quantli:")
            print(df.num_tokens_bert.quantile(0.95))
        return None

    name = 'rcds/swiss_citation_extraction'
    splits = ['train', 'validation', 'test']
    columns = ['text']
    ds = get_dataset_per_lang(name, splits, "")
    print(ds['de'])
    de_df = pd.DataFrame(ds['de'])
    fr_df = pd.DataFrame(ds['fr'])
    it_df = pd.DataFrame(ds['it'])
    df = pd.concat([de_df, fr_df, it_df])
    print(len(df))

    def get_tok(row):
        length = len(row['considerations'])
        return length

    df['tokens'] = df.apply(get_tok, axis='columns')
    print("Mean token length:")
    print(df.loc[:, 'tokens'].mean())
    print("95% quantli:")
    print(df.tokens.quantile(0.95))
    """
    """
    dataset = load_dataset("joelito/lextreme", "swiss_law_area_prediction_sub_area_facts")
    print(dataset)
    print(dataset['train'].features)
    print(dataset['train'].features["label"].int2str(6))
    print(dataset['train'].features["label"].int2str(7))
    print(dataset['train'].features["label"].int2str(5))
    print(dataset['train'].features["label"].int2str(12))
    print(dataset['validation'].features)
    print(dataset['test'].features)
    dataset_6 = dataset.filter(lambda row: row['label'] == 6)
    print(dataset_6[0])
    dataset_7 = dataset.filter(lambda row: row['label'] == 7)
    print(dataset_7[0])
    dataset_5 = dataset.filter(lambda row: row['label'] == 5)
    print(dataset_5[0])
    dataset_12 = dataset.filter(lambda row: row['label'] == 12)
    print(dataset_12[0])
    

    dataset = load_dataset("rcds/swiss_rulings")['train']
    print(dataset)
    dataset = get_rid_of_unused_cols(dataset, ['full_text_num_tokens_bert'])
    df = pd.DataFrame(dataset)
    print("Mean token length:")
    print(df.loc[:, 'full_text_num_tokens_bert'].mean())
    print(df.full_text_num_tokens_bert.describe())
    print("95% quantli:")
    print(df.full_text_num_tokens_bert.quantile(0.95))
    """

    dataset = load_dataset("rcds/swiss_criticality_prediction")
    print(dataset)
    dataset = dataset.filter(lambda example: len(example['facts']) < 1)
    print(dataset)
    dataset = dataset.filter(lambda example: len(example['considerations']) < 1)
    print(dataset)
    dataset = dataset.filter(lambda example: len(example['rulings']) < 1)
    print(dataset)

    dataset = load_dataset("rcds/swiss_leading_decisions")
    print(dataset)
    dataset = dataset.filter(lambda example: len(example['facts']) < 1)
    print(dataset)
    dataset = dataset.filter(lambda example: len(example['considerations']) < 1)
    print(dataset)
    dataset = dataset.filter(lambda example: len(example['rulings']) < 1)
    print(dataset)

    dataset = load_dataset("rcds/swiss_doc2doc_ir")
    print(dataset)
    dataset = dataset.filter(lambda example: len(example['facts']) < 1)
    print(dataset)
    dataset = dataset.filter(lambda example: len(example['considerations']) < 1)
    print(dataset)
    dataset = dataset.filter(lambda example: len(example['rulings']) < 1)
    print(dataset)
