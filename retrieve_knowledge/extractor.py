# -*- coding: utf-8 -*-
import nltk
import string

from nltk.corpus import wordnet as wn
from typing import Text
from transformers import GPT2Tokenizer


class KnowledgeExtractor:

    def __init__(
            self,
            wn_concept_info_path: Text,
            wn_rdf_path: Text,
            tokenizer_type: Text = 'gpt2-medium',
            download_file: bool = False
    ):
        """
        Args
            vocab_path: vocabulary file path for tokenizer
            wn_concept_info_path: WordNet concept information file path
            wn_rdf_path: WordNet h-r-t triplets path
            download_file: whether to download necesary nltk files
        """
        if download_file:
            self._download_nltk()

        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_type)
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.stopwords = set(nltk.corpus.stopwords.words('english'))

        wn_dicts = self.build_WordNet_dict(
            concept_info_path=wn_concept_info_path,
            relation_info_path=wn_rdf_path
        )
        self.offset_to_wn18name_dict = wn_dicts['offset_to_wn18name_dict']
        self.wn18_rdf_dict = wn_dicts['wn18_rdf_dict']

    def process_wn(
            self,
            text: Text,
            filter_stopwords: bool = True,
            ignore_length: int = 0
    ):
        token2synset = dict()
        token2triplets = dict()

        tokens = self.tokenizer.tokenize(text)
        # indexs = tokenizer.convert_tokens_to_ids(tokens)

        for token in set(tokens):
            token = self.tokenizer.convert_tokens_to_string(token).strip()
            if token in set(string.punctuation):
                continue
            if filter_stopwords and token in self.stopwords:
                continue
            if ignore_length > 0 and len(token) <= ignore_length:
                continue

            synsets = wn.synsets(token)
            wn18synset_names = list()
            triplets = list()
            for synset in synsets:
                offset_str = str(synset.offset()).zfill(8)
                if offset_str in self.offset_to_wn18name_dict:
                    wn18synset_names.append(self.offset_to_wn18name_dict[offset_str])
                    # triplets.append(self.wn18_rdf_dict[self.offset_to_wn18name_dict[offset_str]])
            if len(wn18synset_names) > 0:
                token2synset[token] = wn18synset_names
            # if len(triplets) > 0:
                # token2triplets[token] = triplets


        outp = {
            "token2synset": token2synset,
            # "token2triplets": token2triplets
        }
        return outp

    def build_WordNet_dict(self, concept_info_path: Text, relation_info_path: Text):
        # build offest to wn name dict
        offset_to_wn18name_dict = {}
        fin = open(concept_info_path)
        for line in fin:
            info = line.strip().split('\t')
            offset_str, synset_name = info[0], info[1]
            offset_to_wn18name_dict[offset_str] = synset_name

        # build rdf dict
        wn18_rdf_dict = {}
        fin = open(relation_info_path)
        for line in fin:
            info = line.strip().split('\t')
            head, rel, tail = info[0], info[1], info[2]
            wn18_rdf_dict[offset_to_wn18name_dict[info[0]]] = {
                "h": offset_to_wn18name_dict[info[0]],
                "r": rel,
                "t": offset_to_wn18name_dict[info[2]]
            }

        outp = {
            "offset_to_wn18name_dict": offset_to_wn18name_dict,
            "wn18_rdf_dict": wn18_rdf_dict
        }

        return outp

    @staticmethod
    def _download_nltk():
        nltk.download('stopwords')
        nltk.download('wordnet')


# extractor = KnowledgeExtractor(
#     "retrieve_knowledge/wordnet-mlj12-definitions.txt",
#     "retrieve_knowledge/wordnet-mlj12-train.txt",
# )
#
# outp = extractor.process_wn('I want to go to the hospital')
# outp['token2synset']
