# -*- coding: utf-8 -*-

from vocab import Vocab

import codecs
import glob
import pickle

DATA_PATH = "../../NLP/data/text/"
LABEL_DICT = {0:"dokujo-tsushin",
              1:"it-life-hack",
              2:"kaden-channel",
              3:"livedoor-homme",
              4:"movie-enter",
              5:"peachy",
              6:"smax",
              7:"sports-watch",
              8:"topic-news"}

if __name__ == "__main__":

    data_dict = {}
    for label in LABEL_DICT.values():
        files = glob.glob(DATA_PATH+label+"/*")
        docs = []
        for path in files:
            text_list = codecs.open(path, "r","utf-8").readlines()[2:]
            text_list = [txt.replace("\n","") for txt in text_list if not txt == "\n"]
            docs.extend(text_list)
        data_dict[label] = docs


    # leaning vocab
    docs_all = []
    for k,v in data_dict.items():
        docs_all.extend(v)
    vocab = Vocab()
    vocab.set_dict(docs_all)
    vocab.save("./model/news_vocab.pkl")
    #vocab = Vocab.load("./model/news_vocab.pkl")

    # get traindata
    text = text_list[0]
    rev_lab_dict = {v:k for k,v in LABEL_DICT.items()}
    train_dict = {"x":[],"t":[]}
    for k,v in data_dict.items():
        t = rev_lab_dict[k]
        lab = [1 if i == t else 0  for i in range(9)]
        for text in v:
            wakati = Vocab.separate_by_mecab(text)
            ids = vocab.doc2ids(wakati, meta=True)
            train_dict["t"].append(lab)
            train_dict["x"].append(ids)
    # save
    with open("./model/train_data.pkl", mode='wb') as f:
        pickle.dump(train_dict, f)
