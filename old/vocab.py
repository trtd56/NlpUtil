# -*- coding: utf-8 -*-

from collections import defaultdict
import MeCab
import mojimoji
import pickle

class Vocab():
    __UNK_STR  = "<unk>"
    __HEAD_STR = "<s>"
    __TAIL_STR = "</s>"
    __UNK_NUM  = 0
    __HEAD_NUM = 1
    __TAIL_NUM = 2

    def __init__(self):
        self.__stoi = None
        self.__itos = None
        self.__min_count = None

    def set_dict(self, docs, min_count=1):
        self.__min_count = min_count
        wakati_list = [Vocab.separate_by_mecab(text) for text in docs]
        words = self.__grep_words(wakati_list)
        self.__stoi, self.__itos = self.__make_word_dict(words, min_count)

    @staticmethod
    def normalize_document(document):
        document = str(document)
        document = mojimoji.zen_to_han(document, kana=False)
        document = mojimoji.han_to_zen(document, digit=False, ascii=False)
        document = document.lower()
        document = document.replace("\n", " ")
        document = document.replace("\t", " ")
        document = document.replace("\r", " ")
        return document

    @staticmethod
    def separate_by_mecab(document, tagger="-Ochasen"):
        wakati = MeCab.Tagger(tagger)
        document = Vocab.normalize_document(document)
        document_parsed = wakati.parse(document)
        doc = [i.split("\t")[0].strip() for i in document_parsed.split(u"\n")]
        doc.remove(u"EOS")
        doc.remove(u"")
        return doc

    def __grep_words(self, wakati_docs):
        words = []
        for wakati in wakati_docs:
            words.extend(wakati)
        return words

    def __make_word_dict(self, words, min_count):
        # check word frequency
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        word_freq = { word:freq for word, freq in word_freq.items() if freq >= min_count}

        # make sting to int dictionary
        stoi = defaultdict(int)
        # add meta string
        stoi[self.__UNK_STR]  = self.__UNK_NUM
        stoi[self.__HEAD_STR] = self.__HEAD_NUM
        stoi[self.__TAIL_STR] = self.__TAIL_NUM
        for i, (k, v) in enumerate(sorted(word_freq.items(), key=lambda x: -x[1])):
            stoi[k] = i + 3
        stoi = dict(stoi)

        # make int to string dictionary
        itos = {i:s for s,i in stoi.items()}

        return stoi, itos

    def __len__(self):
        return len(self.__stoi.keys())

    def id2word(self, i):
        return self.__itos[i]

    def word2id(self, s):
        try:
            return self.__stoi[s]
        except KeyError:
            return self.__stoi[self.__UNK_STR]

    def ids2doc(self, ids, meta=False):
        doc = [self.id2word(i) for i in ids]
        if not meta:
            return doc
        return [self.__HEAD_STR] + doc + [self.__TAIL_STR]

    def doc2ids(self, doc, meta=False):
        ids = [self.word2id(word) for word in doc]
        if not meta:
            return ids
        return [self.__HEAD_NUM] + ids + [self.__TAIL_NUM]

    def remove_meta(self, data):
        is_head = data[0] == self.__HEAD_NUM or data[0] == self.__HEAD_STR
        is_tail = data[-1] == self.__TAIL_NUM or data[-1] == self.__TAIL_STR
        if is_tail and is_tail:
            return data[1:-1]
        else:
            raise ValueError

    def save(self, path):
        with open(path, mode='wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, mode='rb') as f:
            vocab = pickle.load(f)
        return vocab
