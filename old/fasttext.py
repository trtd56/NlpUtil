# -*- coding: utf-8 -*-

import numpy as np
import subprocess
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from vocab import Vocab
import pickle
from sklearn import cross_validation

IN_PATH = "../../NLP/data/jawiki-abs-wakati.txt"
OUT_DIR = "./model"
DIM = 200
RATIO = 0.1

class FastText():
    """
    cf: https://github.com/facebookresearch/fastText
    """

    def __init__(self, out_dir, dim=200):
        self.in_path = None
        self.out_path = out_dir + "/fasttext_" + str(dim)
        self.__dim = dim

    def supervised(self, wakati, min_count=5):
        self.min_count = min_count
        self.in_path = wakati
        cmd = self.__gen_cmd()
        # TODO: not secure.
        subprocess.call(cmd, shell=True)

    def __gen_cmd(self):
        cmd = "fasttext skipgram -input {} -output {} -dim {} -minCount {}".format(self.in_path, self.out_path, self.__dim, self.min_count)
        return cmd

    def load(self):
        vec_path = self.out_path + ".vec"
        self.__model = self.__load_fasttext_vec(vec_path)

    def __load_fasttext_vec(self, path):
        vectors = {}
        with open(path, "r", encoding="utf-8") as vec:
            for i, line in enumerate(vec):
                try:
                    elements = line.strip().split()
                    word = elements[0]
                    vec = np.array(elements[1:], dtype=float)
                    if not (word in vectors) and len(vec) >= 100:
                        # ignore the case that vector size is invalid
                        vectors[word] = vec
                except ValueError:
                    continue
                except UnicodeDecodeError:
                    continue
            return vectors

    def get_vector(self, word):
        try:
            return self.__model[word]
        except KeyError:
            return np.zeros(self.__dim)

    def get_vec_size(self):
        return self.__dim

    def get_doc_vector(self, doc):
        if type(doc) is not list:
            doc = Vocab.separate_by_mecab(doc)
        vec = np.array([self.get_vector(word) for word in doc])
        if len(vec) == 0:
            return np.zeros(self.__dim)
        return vec.mean(axis=0)

def put_one_label(labels, no):
    t_list = [lab[no] for lab in labels]
    return  np.array(t_list)

if __name__ == '__main__':
    fasttext = FastText(OUT_DIR)
    #fasttext.supervised(IN_PATH)
    fasttext.load()

    vocab = Vocab.load("./model/news_vocab.pkl")
    with open("./model/train_data.pkl", mode='rb') as f:
        train_data = pickle.load(f)
    train_x, vali_x = cross_validation.train_test_split(train_data["x"], test_size=RATIO)
    #train_t, vali_t = cross_validation.train_test_split(train_data["t"], test_size=RATIO)
    train_t, vali_t = cross_validation.train_test_split([t.index(1) for t in train_data["t"]], test_size=RATIO)
    x_vocab = [vocab.remove_meta(vocab.ids2doc(x)) for x in train_x]
    x_vocab_v = [vocab.remove_meta(vocab.ids2doc(x)) for x in vali_x]

    train_features = [fasttext.get_doc_vector(x) for x in x_vocab]
    test_features = [fasttext.get_doc_vector(x) for x in x_vocab_v]
    train_labels = np.array(train_t)
    test_labels = np.array(vali_t)

    #for i in range(9):
    #    train_labels = put_one_label(train_t, i)
    #    test_labels = put_one_label(vali_t, i)

    clf = svm.SVC()
    clf.fit(train_features, train_labels)
    test_pred = clf.predict(test_features)

    print(classification_report(test_labels, test_pred))
    print(accuracy_score(test_labels, test_pred))
