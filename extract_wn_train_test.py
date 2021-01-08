import os
import sys
import argparse
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd

from retrieve_knowledge.extractor import KnowledgeExtractor


def extract_kg_from_sents(sents: list):
    outp = []
    for s in tqdm(sents):
        outp.append(extractor.process_wn(s))
    return outp


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',
                        type=str,
                        default='snli')

    args = parser.parse_args()

    extractor = KnowledgeExtractor(
        "retrieve_knowledge/wordnet-mlj12-definitions.txt",
        "retrieve_knowledge/wordnet-mlj12-train.txt",
    )

    if args.dataset == 'snli':
        data_labels = ['entailment', 'neutral', 'contradiction']

        for data_label in data_labels:
            path = f"dataset_snli/{data_label}"

            print(f"Extracting Knowledge from train dataset: {data_label}")
            df = pd.read_csv(os.path.join(path, 'train.tsv'), sep='\t', encoding='utf-8')
            inputs = df['input'].tolist()
            result = extract_kg_from_sents(inputs)

            with open(os.path.join(path, 'train_wn.dict'), 'wb') as saveFile:
                pickle.dump(result, saveFile)
            del result

            print(f"Extracting Knowledge from dev dataset: {data_label}")
            df = pd.read_csv(os.path.join(path, 'dev.tsv'), sep='\t', encoding='utf-8')
            inputs = df['input'].tolist()
            result = extract_kg_from_sents(inputs)

            with open(os.path.join(path, 'dev_wn.dict'), 'wb') as saveFile:
                pickle.dump(result, saveFile)
            del result

            print(f"Extracting Knowledge from test dataset: {data_label}")
            df = pd.read_csv(os.path.join(path, 'test.tsv'), sep='\t', encoding='utf-8')
            inputs = df['input'].tolist()
            result = extract_kg_from_sents(inputs)

            with open(os.path.join(path, 'test_wn.dict'), 'wb') as saveFile:
                pickle.dump(result, saveFile)
            del result
