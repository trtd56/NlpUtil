# -*- coding: utf-8 -*-

import datetime
import pandas as pd
from chainer import optimizer, optimizers, serializers

from att_s2s import GrobalAttention
from xp import XP
from vocab import Vocab

PATH = "../../NLP/data/basic_sentence.csv"

def gen_batch(ja, en, size):
    x_batch = []
    t_batch = []
    for j, e in zip(ja, en):
        j = Vocab.wakati_docs([j])[0]
        e = "".join([i for i in e if i != "."])
        e = Vocab.normalize_document(e).split()
        x_id = XP.iarray(ja_vocab.doc2ids(j, meta=True))
        t_id = XP.iarray(en_vocab.doc2ids(e, meta=True))
        if len(x_batch) <= size:
            x_batch.append(x_id)
            t_batch.append(t_id)
        else:
            yield x_batch, t_batch
            x_batch = []
            t_batch = []
    if len(x_batch) > 0:
        yield x_batch, t_batch

if __name__ == "__main__":
    gpu = 7
    n_batch = 20
    min_count = 1

    # load text
    print("load text")
    data = pd.read_csv(PATH, header=None)
    ja = data[0].tolist()[:]
    en = data[1].tolist()[:]
    ja_vocab = Vocab()
    en_vocab = Vocab()
    ja_vocab.set_dict(doc_list=ja, min_count=min_count)
    en_vocab.set_dict(doc_list=en, min_count=min_count)

    XP.set_library(gpu=gpu)

    model = GrobalAttention(1, 200, len(ja_vocab), len(en_vocab))
    if gpu >= 0:
        model.to_gpu(gpu)

    opt = optimizers.RMSprop(lr=0.0005)
    opt.setup(model)
    opt.add_hook(optimizer.GradientClipping(5))

    # evaluation data
    ja_v = ja[0]
    en_v = en[0]
    ja_v_list = Vocab.wakati_docs([ja_v])[0]
    v_id = XP.iarray(ja_vocab.doc2ids(ja_v_list, meta=True))

    print("start training")
    for epoch in range(100):
        count = 0
        for j, e in gen_batch(ja, en, n_batch):
            model.cleargrads()
            loss = model(j, e, True)
            loss.backward()
            opt.update()
            loss.unchain_backward()
            count += n_batch
            now = datetime.datetime.now()
            print("{}\tepoch:{}\tbatch:{}/{}\tloss:{}".format(now,epoch,count,len(ja),loss.data))
        att_w, h_list = model.calc_att_w(v_id)
        print("ja :",ja_v)
        print("en :",en_v)
        print("hyp:"," ".join(en_vocab.ids2doc(h_list)))
    # check attention weight
    hyp = en_vocab.ids2doc(h_list)
    print(Vocab.wakati_docs([ja_v])[0])
    print("att")
    for a, h in zip(att_w, hyp):
        print(h)
        print(a.data[0])
    print("save model")
    # save model
    model.to_cpu()
    serializers.save_npz('./model/att_model_01.npz', model)
    ja_vocab.save("./model/ja_vocab.pkl")
    en_vocab.save("./model/en_vocab.pkl")
