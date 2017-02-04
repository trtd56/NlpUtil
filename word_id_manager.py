# -*- coding: utf-8 -*-

from vocab import Vocab
from xp import XP

import pickle
from sklearn.cross_validation import train_test_split

class WordIdManager():

    __IGNORE_LABEL = -1

    def __init__(self, doc_list, lab_list):
        self.doc_list = doc_list
        self.lab_dict = {v:i for i,v in enumerate(set(lab_list))}
        self.lab_list = [self.lab_dict[i] for i in lab_list]

    def set_data(self, min_count, raito, max_words, gpu):
        self.__get_words_ids(min_count)
        self.__split_data(raito)
        self.__trans_xp(gpu, max_words)

    def __get_words_ids(self, min_count):
        self.vocab = Vocab()
        self.vocab.set_dict(self.doc_list, min_count)
        self.ids = [self.vocab.doc2ids(Vocab.separate_by_mecab(text), meta=True) for text in self.doc_list]

    def __split_data(self, raito):
        self.train_x, self.vali_x = train_test_split(self.ids, test_size=raito)
        self.train_t, self.vali_t = train_test_split(self.lab_list, test_size=raito)

    def __fill_batch(self, data, max_words):
        max_len = max([len(x) for x in data])
        data2 = [x + [self.__IGNORE_LABEL] * (max_len - len(x)) for x in data]
        return [x[:max_words] for x in data2]

    def __trans_xp(self, gpu, max_words):
        XP.set_library(gpu)
        self.train_x = XP.iarray(self.__fill_batch(self.train_x, max_words)).data
        self.train_t = XP.iarray(self.train_t).data
        self.vali_x  = XP.iarray(self.__fill_batch(self.vali_x, max_words)).data
        self.vali_t  = XP.iarray(self.vali_t).data

    def get_data(self):
         return self.train_x, self.train_t, self.vali_x, self.vali_t

    def get_n_vocab(self):
        return len(self.vocab)

    def get_n_label(self):
        return len(set(self.lab_list))

    def get_label_dict(self):
        return {v:k for k,v in self.lab_dict.items()}

    def save(self, path):
        with open(path, mode='wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, mode='rb') as f:
            manager = pickle.load(f)
        return manager
