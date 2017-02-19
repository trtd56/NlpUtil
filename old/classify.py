# -*- coding: utf-8 -*-

import chainer.links as L
import chainer.functions as F
from xp import XP
from vocab import Vocab

from chainer import Chain, optimizer, optimizers, training, iterators
from chainer.training import extensions
from chainer.datasets import tuple_dataset
import pickle
from sklearn import cross_validation

from att_s2s import NStepLSTM
PATH = "../../NLP/data/basic_sentence.csv"
MAX_WORDS = 60


class SimpleLstmNet(Chain):

    def __init__(self, n_layers, n_unit, n_vocab, n_out):
        use_cudnn = (gpu >= 0)
        super(SimpleLstmNet, self).__init__(
            embed = L.EmbedID(n_vocab, n_unit, ignore_label=-1),
            nstep_lstm = NStepLSTM(n_layers, n_unit, n_unit),
            l1 = L.Linear(n_unit, n_out),
        )

    def __call__(self, x, train=True):
        h_x = self.embed(x)
        h_x = [h for h in h_x]
        self.nstep_lstm.reset_state()
        h_x = self.nstep_lstm(h_x, train)
        h_x = [h[-1] for h in h_x]
        h_x = F.stack(h_x, 0)
        return self.l1(F.dropout(h_x, train=train))

class SimpleCnnNet(Chain):

    def __init__(self, output_channel, filter_height, filter_width, n_unit, n_out, n_vocab, gpu):
        use_cudnn = (gpu >= 0)
        self.vec_size = filter_width
        super(SimpleCnnNet, self).__init__(
            embed = L.EmbedID(n_vocab, filter_width, ignore_label=-1),
            cnn1 = L.Convolution2D(1, output_channel, (filter_height, filter_width), use_cudnn=use_cudnn),
            bnorm1=L.BatchNormalization(output_channel),
            l1 = L.Linear(output_channel, n_unit),
            l2 = L.Linear(n_unit, n_out),
        )

    def __call__(self, x):
        h_x = self.embed(x)
        n_words = h_x.shape[1]
        h_x = F.expand_dims(h_x, 1)
        h_x = F.relu(self.bnorm1(self.cnn1(h_x)))
        h_x = F.max_pooling_2d(h_x, (n_words, self.vec_size))
        h_x = F.relu(self.l1(h_x))
        return self.l2(h_x)

def fill_batch(data):
    max_len = max([len(x) for x in data])
    data2 = [x + [-1] * (max_len - len(x)) if not len(x) == max_len else x for x in data]
    return [x[:MAX_WORDS] for x in data2]

if __name__ == "__main__":
    GPU = 6
    N_EPOCH = 20
    BATCH = 100
    RATIO = 0.05
    VEC_SIZE = 300
    XP.set_library(gpu=GPU)

    print("load data")
    vocab = Vocab.load("./model/news_vocab.pkl")
    with open("./model/train_data.pkl", mode='rb') as f:
        train_data = pickle.load(f)

    print("divide data")
    train_x, vali_x = cross_validation.train_test_split(train_data["x"], test_size=RATIO)
    #train_t, vali_t = cross_validation.train_test_split(train_data["t"], test_size=RATIO)
    train_t, vali_t = cross_validation.train_test_split([t.index(1) for t in train_data["t"]], test_size=RATIO)

    xp_train_x = XP.iarray(fill_batch(train_x)).data
    xp_train_t = XP.iarray(train_t).data
    xp_vali_x = XP.iarray(fill_batch(vali_x)).data
    xp_vali_t = XP.iarray(vali_t).data

    train = tuple_dataset.TupleDataset(xp_train_x, xp_train_t)
    test = tuple_dataset.TupleDataset(xp_vali_x, xp_vali_t)
    train_iter = iterators.SerialIterator(train, BATCH)
    test_iter = iterators.SerialIterator(test, BATCH, repeat=False, shuffle=False)

    #model = L.Classifier(SimpleCnnNet(10, 3, VEC_SIZE, 30, len(vali_t[0]), len(vocab), GPU), lossfun=F.sigmoid_cross_entropy)
    model = L.Classifier(SimpleCnnNet(30, 3, VEC_SIZE, 200, 9, len(vocab), GPU), lossfun=F.softmax_cross_entropy)
    #model = L.Classifier(SimpleLstmNet(2, VEC_SIZE, len(vocab), len(vali_t[0])), lossfun=F.sigmoid_cross_entropy)
    #model = L.Classifier(SimpleLstmNet(2, VEC_SIZE, len(vocab), 9, lossfun=F.softmax_cross_entropy)

    #model.compute_accuracy = False
    opt = optimizers.RMSprop(lr=0.005)
    opt.setup(model)
    opt.add_hook(optimizer.GradientClipping(5))

    updater = training.StandardUpdater(train_iter, opt, device=GPU)
    trainer = training.Trainer(updater, (N_EPOCH, 'epoch'), out="result")

    trainer.extend(extensions.Evaluator(test_iter, model, device=GPU))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(N_EPOCH, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))
    #trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()

    BASE = 0.5
    count = 0
    for x, t in zip(fill_batch(vali_x), vali_t):
        x = XP.iarray([x]).data
        pred = F.sigmoid(model.predictor(x, train=False)).data.tolist()[0]
        hyp = [1 if i > BASE else 0 for i in pred]
        ans = [1 if i > BASE else 0 for i in t]
        if hyp == ans:
            count += 1
    print("acc: {}".format(count/len(vali_t)))
