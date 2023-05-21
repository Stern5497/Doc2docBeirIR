import datasets
import wandb
import json
import os, pathlib
import ast
import random
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from transformers import AutoTokenizer


if __name__ == '__main__':
    print("Start")
    dataset = datasets.load_dataset('joelito/lextreme', 'swiss_judgment_prediction_xl_considerations')
    print(dataset)