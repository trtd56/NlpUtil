# -*- coding: utf-8 -*-

import pandas as pd

from word_id_manager import WordIdManager
from net import SimpleCnnNet
from trainer import SoftMaxTrainer

# common
gpu     = -1
out_dir = "result"
# WordIdManager
min_count = 1
raito     = 0.1
max_words = 20
# SimpleCnnNet
output_channel = 10
filter_height  = 3
filter_width   = 100
n_unit         = 50
# SoftMaxTrainer
n_batch = 2
n_epoch = 10
g_clip  = 2
opt     = "RMSprop"
lr      = 0.005

if __name__ == "__main__":

    df = pd.read_csv("./sample.csv", header=None)
    doc_list = df[1].tolist()
    lab_list = df[0].tolist()

    # get train data
    manager = WordIdManager(doc_list, lab_list)
    manager.set_data(min_count, raito, max_words, gpu)
    train_x, train_t, valid_x, valid_t = manager.get_data()
    n_out   = manager.get_n_label()
    n_vocab = manager.get_n_vocab()

    # set learning model
    net = SimpleCnnNet(output_channel, filter_height, filter_width, n_unit, n_out, n_vocab, gpu)
    trainer = SoftMaxTrainer(net)
    trainer.set_train_data(train_x, train_t, valid_x, valid_t, n_batch)
    trainer.set_trainer(out_dir, gpu, n_epoch, g_clip, opt, lr)

    # learning start
    trainer.start()

    # predict
    #import numpy as np
    #sample = np.array([train_x[0]],dtype=np.int32)
    #pred = trainer.predict(sample)
    #lab_dict = manager.get_label_dict()
    #print(lab_dict[pred.argmax(1)[0]])
