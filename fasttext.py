# -*- coding: utf-8 -*-

import subprocess
import pandas as pd
from sklearn.cross_validation import train_test_split

from vocab import Vocab

class FastText():
    """
    cf: https://github.com/facebookresearch/fastText
    """

    __TRAIN_FILE = "wakati.txt"
    __MODEL_FILE = "model"
    __TEST_FILE  = "wakati_vali.txt"

    def __init__(self, model_dir):
        self.model_dir   = model_dir
        self.wakati_path = "{}/{}".format(model_dir, self.__TRAIN_FILE)
        self.model_path  = "{}/{}".format(model_dir, self.__MODEL_FILE)
        self.test_path  = "{}/{}".format(model_dir, self.__TEST_FILE)
        self.__init_param()

    def set_train_data(self, doc_list, lab_list, raito):
        train_x, test_x = train_test_split(doc_list, test_size=raito)
        train_t, test_t = train_test_split(lab_list, test_size=raito)
        self.make_text_file(self.wakati_path, train_x, train_t)
        self.make_text_file(self.test_path, test_x, test_t)

    def make_text_file(self, path, doc_list, lab_list):
        with open(path, mode='w') as f:
            for doc, lab in zip(doc_list, lab_list):
                doc = " ".join(Vocab.separate_by_mecab(doc))
                line = "__label__{} , {}\n".format(lab, doc)
                f.writelines(line)

    def supervised(self):
        cmd = ["fasttext", "supervised", "-input", self.wakati_path, "-output", self.model_path]
        cmd += self.__get_optional_param()
        cmd = [str(i) for i in cmd]
        subprocess.check_call(cmd)

    def test(self, k=1):
        cmd = ["fasttext", "test", self.model_path+".bin", self.test_path, k]
        cmd = [str(i) for i in cmd]
        subprocess.check_call(cmd)

    def predict(self, k=1):
        cmd = ["fasttext", "predict", self.model_path+".bin", self.test_path, k]
        cmd = [str(i) for i in cmd]
        subprocess.check_call(cmd)

    def set_param(self, param_dict):
        for k,v in param_dict.items():
            exec("self.{} = {}".format(k, v))

    def __init_param(self):
        self.lr                 = 0.1           #learning rate [0.1]
        self.lrUpdateRate       = 100           #change the rate of updates for the learning rate [100]
        self.dim                = 100           #size of word vectors [100]
        self.ws                 = 5             #size of the context window [5]
        self.epoch              = 5             #number of epochs [5]
        self.minCount           = 1             #minimal number of word occurences [1]
        self.minCountLabel      = 0             #minimal number of label occurences [0]
        self.neg                = 5             #number of negatives sampled [5]
        self.wordNgrams         = 1             #max length of word ngram [1]
        self.loss               = "ns"          #loss function {ns, hs, softmax} [ns]
        self.bucket             = 2000000       #number of buckets [2000000]
        self.minn               = 0             #min length of char ngram [0]
        self.maxn               = 0             #max length of char ngram [0]
        self.thread             = 12            #number of threads [12]
        self.t                  = 0.0001        #sampling threshold [0.0001]
        self.label              = "__label__"   #labels prefix [__label__]
        self.verbose            = 2             #verbosity level [2]
        self.pretrainedVectors  = None          #pretrained word vectors for supervised learning []

    def __get_optional_param(self):
        params = ["-lr",self.lr,"-lrUpdateRate",self.lrUpdateRate,"-dim",self.dim,
                  "-ws",self.ws,"-epoch",self.epoch,"-minCount",self.minCount,
                  "-minCountLabel",self.minCountLabel,"-neg",self.neg,"-wordNgrams",self.wordNgrams,
                  "-loss",self.loss,"-bucket",self.bucket,"-minn",self.minn,"-maxn",self.maxn,
                  "-thread",self.thread,"-t",self.t,"-label",self.label,"-verbose",self.verbose]
        if not self.pretrainedVectors:
            params + ["-pretrainedVectors",self.pretrainedVectors]
        return params

if __name__ == '__main__':
    df = pd.read_csv("./sample.csv", header=None)
    doc_list = df[1].tolist()
    lab_list = df[0].tolist()
    raito = 0.1

    fasttext = FastText("./result")
    fasttext.set_train_data(doc_list, lab_list, raito)

    param_dict = {"dim":150, "epoch":10}
    fasttext.set_param(param_dict)

    fasttext.supervised()
    fasttext.test()
    fasttext.predict()
