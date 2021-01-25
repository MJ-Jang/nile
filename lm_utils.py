import pandas as pd
import os
import pickle
import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset
from retrieve_knowledge.utils import loadRotatEModel
from retrieve_knowledge.extractor import KnowledgeExtractor

EXP_TOKEN = '[EXP]'
EOS_TOKEN = '[EOS]'


class TSVDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train', block_size=512, get_annotations=False):
        self.print_count = 5
        self.eos_token_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)

        cached_features_file, data = self.load_data(file_path, block_size, args)
        self.data = data

        if get_annotations: cached_features_file = cached_features_file + '_annotated'

        # if os.path.exists(cached_features_file):
        #     print ('Loading features from', cached_features_file)
        #     with open(cached_features_file, 'rb') as handle:
        #         self.examples = pickle.load(handle)
        #     return

        print ('Saving features from ', file_path, ' into ', cached_features_file) 

        def create_example(r):
            text1 = '{} {} '.format(r['input'], EXP_TOKEN)
            tokenized_text1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text1))
            prompt_length = len(tokenized_text1)
            tokenized_text, total_length = tokenized_text1, len(tokenized_text1)
            if get_annotations:
                text2 = r['target']
                tokenized_text2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text2))
                tokenized_text = tokenized_text1 + tokenized_text2
                tokenized_text = tokenized_text + [self.eos_token_id]
                total_length = len(tokenized_text)
                if len(tokenized_text) > block_size:
                    tokenized_text = tokenized_text[:block_size]
                if len(tokenized_text) < block_size:
                    tokenized_text = tokenized_text + [self.eos_token_id] * (block_size-len(tokenized_text))
            if self.print_count > 0:
                print ('example: ', text1 + text2 if get_annotations else text1)
                self.print_count = self.print_count - 1
            return (tokenized_text, prompt_length, total_length)

        self.examples = data.apply(create_example, axis=1).to_list()

        print ('Saving ', len(self.examples), ' examples')
        with open(cached_features_file, 'wb') as handle:
            pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item][0]), self.examples[item][1], self.examples[item][2]

    def get_example_text(self, index):
        return self.data['prompt'][index]

    def add_explanation(self, index, explanation):
        explanation_name = 'Generated_Explanation'
        self.data.at[self.data.index[index], explanation_name] = explanation

    def load_data(self, file_path, block_size, args):
        assert os.path.isfile(file_path)
        data = pd.read_csv(file_path, sep='\t', index_col='pairID')
        if args.train_ratio < 1.0:
            data = data.sample(frac=args.train_ratio, random_state=args.seed).reset_index(drop=True)
        print (data)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, 'cached_lm_{}_{}'.format(block_size, filename))
        return cached_features_file, data

    def save(self, filename):
        self.data.to_csv(filename, sep='\t')


class KGTSVDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train',
                 entity_vec_path=None, block_size=512, get_annotations=False):
        self.print_count = 5
        self.eos_token_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)

        # entity_vec_path = 'retrieve_knowledge/RotatE_WN18_512d.txt'
        entity_embeddings = loadRotatEModel(entity_vec_path) # add to the argument
        self.entity_dim = entity_embeddings[list(entity_embeddings.keys())[0]].shape[0]

        extractor = KnowledgeExtractor(
            "retrieve_knowledge/wordnet-mlj12-definitions.txt",
            "retrieve_knowledge/wordnet-mlj12-train.txt",
        ) # hardcoded

        cached_features_file, data = self.load_data(file_path, block_size)
        self.data = data

        if get_annotations: cached_features_file = cached_features_file + '_annotated'

        if os.path.exists(cached_features_file):
            print('Loading features from', cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
            return

        print('Saving features from ', file_path, ' into ', cached_features_file)

        def create_example(r):
            text1 = '{} {} '.format(r['input'], EXP_TOKEN)
            tokenized_text1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text1))
            prompt_length = len(tokenized_text1)
            tokenized_text, total_length = tokenized_text1, len(tokenized_text1)

            # entity embedding
            entity_vec = self._extract_entity_vecs(text1, tokenizer, extractor, entity_embeddings)

            if get_annotations:
                text2 = r['target']
                tokenized_text2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text2))
                tokenized_text = tokenized_text1 + tokenized_text2
                tokenized_text = tokenized_text + [self.eos_token_id]
                total_length = len(tokenized_text)

                entity_vec_2 = self._extract_entity_vecs(text2, tokenizer, extractor, entity_embeddings)
                entity_vec_2 = np.vstack([entity_vec_2, np.array([0]*self.entity_dim)]) # for eos token

                if total_length > block_size:
                    tokenized_text = tokenized_text[:block_size]
                    entity_vec_2 = entity_vec_2[:block_size, :]

                if total_length < block_size:
                    len_diff = block_size - total_length
                    tokenized_text = tokenized_text + [self.eos_token_id] * len_diff
                    pad_vec = np.array([[0] * self.entity_dim] * len_diff)
                    entity_vec_2 = np.vstack([entity_vec_2,pad_vec])

                entity_vec = np.vstack([entity_vec, entity_vec_2])

            if self.print_count > 0:
                print('example: ', text1 + text2 if get_annotations else text1)
                self.print_count = self.print_count - 1
            return (tokenized_text, prompt_length, total_length, entity_vec)

        self.examples = data.apply(create_example, axis=1).to_list()
        print('Saving ', len(self.examples), ' examples')
        with open(cached_features_file, 'wb') as handle:
            pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item][0]), self.examples[item][1], self.examples[item][2], torch.tensor(self.examples[item][3])

    def get_example_text(self, index):
        return self.data['prompt'][index]

    def add_explanation(self, index, explanation):
        explanation_name = 'Generated_Explanation'
        self.data.at[self.data.index[index], explanation_name] = explanation

    def load_data(self, file_path, block_size):
        assert os.path.isfile(file_path)
        data = pd.read_csv(file_path, sep='\t', index_col='pairID')
        print(data)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, 'cached_lm_{}_{}'.format(block_size, filename))
        return cached_features_file, data

    def save(self, filename):
        self.data.to_csv(filename, sep='\t')

    def _extract_entity_vecs(
            self,
            text,
            tokenizer,
            extractor,
            embeddings,
    ):
        tokens = tokenizer.tokenize(text)
        extracted_entities = extractor.process_wn(text)['token2synset']

        entity_vec = []
        if not extracted_entities:
            length = len(tokens)
            return np.array([[0] * self.entity_dim] * length)
        for t in tokens:
            entities = extracted_entities.get(t, None)
            if entities:
                # if multiple entities, take average
                vec_ = np.array([embeddings.get(e) for e in entities]).mean(axis=0)
            else:
                vec_ = np.array([0] * self.entity_dim)
            entity_vec.append(vec_)
        return np.array(entity_vec)




class KGTSVDataset2(Dataset):
    def __init__(self, tokenizer, args, file_path='train',
                 entity_vec_path=None, block_size=512, get_annotations=False):
        self.print_count = 5
        self.eos_token_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)

        # entity_vec_path = 'retrieve_knowledge/RotatE_WN18_512d.txt'
        self.entity_embeddings = loadRotatEModel(entity_vec_path) # add to the argument
        self.entity_dim = self.entity_embeddings[list(self.entity_embeddings.keys())[0]].shape[0]

        self.extractor = KnowledgeExtractor(
            "retrieve_knowledge/wordnet-mlj12-definitions.txt",
            "retrieve_knowledge/wordnet-mlj12-train.txt",
            download_file=True
        ) # hardcoded

        self.tokenizer = tokenizer
        self.get_annotations = get_annotations
        self.block_size = block_size

        cached_features_file, data = self.load_data(file_path, block_size)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        r = self.data.iloc[item]

        def create_example(r):
            text1 = '{} {} '.format(r['input'], EXP_TOKEN)
            tokenized_text1 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text1))
            prompt_length = len(tokenized_text1)
            tokenized_text, total_length = tokenized_text1, len(tokenized_text1)

            # entity embedding
            entity_vec = self._extract_entity_vecs(text1, self.tokenizer, self.extractor, self.entity_embeddings)

            if self.get_annotations:
                text2 = r['target']
                tokenized_text2 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text2))
                tokenized_text = tokenized_text1 + tokenized_text2
                tokenized_text = tokenized_text + [self.eos_token_id]
                total_length = len(tokenized_text)

                entity_vec_2 = self._extract_entity_vecs(text2, self.tokenizer, self.extractor, self.entity_embeddings)
                entity_vec_2 = np.vstack([entity_vec_2, np.array([0]*self.entity_dim)]) # for eos token
                entity_vec = np.vstack([entity_vec, entity_vec_2])

                if total_length > self.block_size:
                    tokenized_text = tokenized_text[:self.block_size]
                    entity_vec = entity_vec[:self.block_size, :]

                if total_length < self.block_size:
                    len_diff = self.block_size - total_length
                    tokenized_text = tokenized_text + [self.eos_token_id] * len_diff

                    pad_vec = np.array([[0] * self.entity_dim] * len_diff)
                    entity_vec = np.vstack([entity_vec, pad_vec])
            return (tokenized_text, prompt_length, total_length, entity_vec)

        tokenized_text, prompt_len, total_len, entity_vec = create_example(r)
        return torch.tensor(tokenized_text), prompt_len, total_len, torch.tensor(entity_vec)

    def get_example_text(self, index):
        return self.data['prompt'][index]

    def add_explanation(self, index, explanation):
        explanation_name = 'Generated_Explanation'
        self.data.at[self.data.index[index], explanation_name] = explanation

    def load_data(self, file_path, block_size):
        assert os.path.isfile(file_path)
        data = pd.read_csv(file_path, sep='\t', index_col='pairID')
        print(data)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, 'cached_lm_{}_{}'.format(block_size, filename))
        return cached_features_file, data

    def save(self, filename):
        self.data.to_csv(filename, sep='\t')

    def _extract_entity_vecs(
            self,
            text,
            tokenizer,
            extractor,
            embeddings,
    ):
        tokens = tokenizer.tokenize(text)
        extracted_entities = extractor.process_wn(text)['token2synset']

        entity_vec = []
        if not extracted_entities:
            length = len(tokens)
            return np.array([[0] * self.entity_dim] * length)
        for t in tokens:
            entities = extracted_entities.get(t, None)
            if entities:
                # if multiple entities, take average
                vec_ = np.array([embeddings.get(e) for e in entities]).mean(axis=0)
            else:
                vec_ = np.array([0] * self.entity_dim)
            entity_vec.append(vec_)
        return np.array(entity_vec)


