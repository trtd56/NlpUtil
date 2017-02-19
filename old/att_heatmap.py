# -*- coding: utf-8 -*-

"""
plot cf http://hacknote.jp/archives/18645/
"""

import matplotlib
matplotlib.use('Agg')

from chainer import serializers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


from xp import XP
from att_s2s import GrobalAttention
from vocab import Vocab

PATH = "../../NLP/data/basic_sentence.csv"
FONT = "../../NLP/data/ipam00303/ipam.ttf"
fp = FontProperties(fname=FONT)

def plot_heatmap(a_list, row_labels, column_labels, img_path=None):

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(a_list, cmap=plt.cm.Blues)

    ax.set_xticks(np.arange(a_list.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(a_list.shape[0])+0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(row_labels, minor=False, fontproperties = fp)
    ax.set_yticklabels(column_labels, minor=False, fontproperties = fp)

    if img_path:
        plt.savefig(img_path)
    else:
        plt.show()

if __name__ == "__main__":

    gpu = 7
    XP.set_library(gpu=gpu)

    data = pd.read_csv(PATH, header=None)
    ja = data[0].tolist()[:]
    en = data[1].tolist()[:]

    ja_vocab = Vocab.load("./model/ja_vocab.pkl")
    en_vocab = Vocab.load("./model/en_vocab.pkl")


    model = GrobalAttention(1, 200, len(ja_vocab), len(en_vocab))
    serializers.load_npz('./model/att_model_01.npz', model)
    if gpu >= 0:
        model.to_gpu(gpu)

    count = 1
    for ja_v, en_v in zip(ja,en):
    # evaluation data
        ja_v_list = Vocab.separate_by_mecab(ja_v)
        v_id = XP.iarray(ja_vocab.doc2ids(ja_v_list, meta=True))

        att_w, h_list = model.calc_att_w(v_id)
        a_list = []
        for a in att_w:
            a_list.append(a.data[0].tolist())

        a_list = np.array(a_list)
        row_lab = ["<s>"] + ja_v_list + ["</s>"]
        col_lab = [en_vocab.id2word(i) for i in h_list]

        img_name = "./heatmap/heatmap_{0:04d}.png".format(count)
        plot_heatmap(a_list, row_lab, col_lab, img_path=img_name)
        count += 1
    #plot_heatmap(a_list, row_lab, col_lab)
