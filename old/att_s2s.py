# -*- coding: utf-8 -*-

"""
cf.
https://github.com/odashi/chainer_examples/blob/master/chainer-1.5/mt_s2s_attention.py
http://qiita.com/halhorn/items/614f8fe1ec7663e04bea
http://www.slideshare.net/YusukeOda1/chainerrnn
http://ksksksks2.hatenadiary.jp/category/%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86
http://www.slideshare.net/yutakikuchi927/deep-learning-nlp-attention

https://talbaumel.github.io/attention/
http://www.slideshare.net/KeonKim/attention-mechanisms-with-tensorflow
https://github.com/farizrahman4u/seq2seq/blob/master/seq2seq/models.py
http://www.slideshare.net/sheemap/convolutional-neural-netwoks
"""

import chainer.links as L
import chainer.functions as F
from chainer import Chain, Variable, optimizer, optimizers, cuda

from xp import XP


class NStepLSTM(L.NStepLSTM):
    """
    NStepLSTMの初期化と呼び出しをL.LSTMに近づけたサブクラス
    """

    def __init__(self, n_layers, in_size, out_size, dropout=0.5, use_cudnn=True):
        super(NStepLSTM, self).__init__(n_layers, in_size, out_size, dropout, use_cudnn)
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(NStepLSTM, self).to_cpu()
        if self.cx is not None:
            self.cx.to_cpu()
        if self.hx is not None:
            self.hx.to_cpu()

    def to_gpu(self, device=None):
        super(NStepLSTM, self).to_gpu(device)
        if self.cx is not None:
            self.cx.to_gpu(device)
        if self.hx is not None:
            self.hx.to_gpu(device)

    def set_state(self, cx, hx):
        assert isinstance(cx, Variable)
        assert isinstance(hx, Variable)
        cx_ = cx
        hx_ = hx
        if self.xp == numpy:
            cx_.to_cpu()
            hx_.to_cpu()
        else:
            cx_.to_gpu()
            hx_.to_gpu()
        self.cx = cx_
        self.hx = hx_

    def reset_state(self):
        self.cx = self.hx = None

    def __call__(self, xs, train=True):
        batch = len(xs)
        if self.hx is None:
            xp = self.xp
            self.hx = Variable(
                xp.zeros((self.n_layers, batch, self.state_size), dtype=xs[0].dtype),
                volatile='auto')
        if self.cx is None:
            xp = self.xp
            self.cx = Variable(
                xp.zeros((self.n_layers, batch, self.state_size), dtype=xs[0].dtype),
                volatile='auto')
        hy, cy, ys = super(NStepLSTM, self).__call__(self.hx, self.cx, xs, train)
        self.hx, self.cx = hy, cy
        return ys

class BiNstepLstm(Chain):

    def __init__(self, n_layers, n_unit, n_vocab):
        super(BiNstepLstm, self).__init__(
            embed = L.EmbedID(n_vocab, n_unit),
            nstep_lstm_f = NStepLSTM(n_layers, n_unit, n_unit),
            nstep_lstm_b = NStepLSTM(n_layers, n_unit, n_unit),
        )

    def reset_state(self):
        self.nstep_lstm_f.reset_state()
        self.nstep_lstm_b.reset_state()

    def __call__(self, x, train):
        x = [self.embed(i) for i in x]
        x_b = [i[::-1] for i in x]
        self.reset_state()
        h_x_f = self.nstep_lstm_f(x, train)
        h_x_b = self.nstep_lstm_b(x_b, train)
        return [F.concat([f, b[::-1]]) for f,b in zip(h_x_f,h_x_b)]

class AttentionNet(Chain):

    __L_LIMIT = 10

    def __init__(self, n_layer, n_unit, n_vocab):
        super(AttentionNet, self).__init__(
            l1 = L.Linear(n_unit, n_unit),
            l2 = L.Linear(n_unit, n_unit),
            fnn = L.Linear(n_unit, 1),
            lstm = L.LSTM(n_unit, n_unit),
            dec = L.Linear(n_unit, n_vocab),
        )

    def reset_state(self):
        self.lstm.reset_state()

    def get_attention(self, h, s_1):
        w1 = self.l1(h)
        w2 = self.l2(s_1)
        v_list = [self.fnn(F.tanh(F.expand_dims(_w1,0)+w2)) for _w1 in w1]
        energy = F.reshape(F.stack(v_list, 0), (1, len(v_list)))
        return F.softmax(energy)

    def __call__(self, h, t=None, train=False):
        self.reset_state()
        if train:
            rep_lim = len(t)
            loss = 0
        else:
            rep_lim = self.__L_LIMIT
            hyp_list = []
            a_list = []
        dim = (1, len(h[0]))
        self.lstm.c = Variable(self.xp.zeros(dim, dtype=self.xp.float32))
        for i in range(rep_lim):
            a = self.get_attention(h, self.lstm.c)
            c = F.matmul(a, h)
            y = self.dec(self.lstm(c))
            if train:
                loss += F.softmax_cross_entropy(y, F.reshape(t[i],(1,)))
            else:
                hyp = cuda.to_cpu(y.data.argmax(1))[0]
                hyp_list.append(hyp)
                a_list.append(a)
                if hyp == 2:
                    break
        if train:
            return loss
        return [a_list, hyp_list]

class GrobalAttention(Chain):

    def __init__(self, n_layer, n_unit, n_enc, n_dec):
        super(GrobalAttention, self).__init__(
            enc = BiNstepLstm(n_layer, n_unit, n_enc),
            att = AttentionNet(n_layer, n_unit*2, n_dec),
        )

    def __call__(self, x, t=None, train=False):

        h = self.enc(x, train)
        if train:
            loss = [self.att(hh, t=tt, train=True) for hh, tt in zip(h, t)]
            return sum(loss)
        else:
            h_x = [self.att(hh, train=False)[1] for hh in h]
            return h_x

    def calc_att_w(self, x):
        h = self.enc([x], False)
        a_list, hyp_list = self.att(h[0], train=False)
        return a_list, hyp_list

if __name__ == "__main__":

    # chainer_variable_converter
    X_IDS = [[1,3,4,2], [1,7,3,9,2],[1,11,7,2]]
    T_IDS = [[1,5,6,2], [1,8,5,10,2],[1,12,8,2]]
    gpu = -1

    XP.set_library(gpu=gpu)
    x_ids = [XP.iarray(i, is_volatile=False) for i in X_IDS]
    t_ids = [XP.iarray(i, is_volatile=False) for i in T_IDS]


    model = GrobalAttention(1, 200, 15, 15)
    if gpu >= 0:
        model.to_gpu(gpu)

    opt = optimizers.RMSprop(lr=0.0005)
    opt.setup(model)
    opt.add_hook(optimizer.GradientClipping(5))

    for epoch in range(100):
        model.cleargrads()
        loss = model(x_ids, t_ids, True)
        loss.backward()
        opt.update()
        loss.unchain_backward()
        print("epoch:",epoch,"/ train loss:",loss.data)
        # check over fitting
        hyp = model(x_ids, train=False)
        print("hyp",[" ".join(map(str,h)) for h in hyp])
        print("ans",[" ".join(map(str,h)) for h in T_IDS])
    # check attention weight
    for i in range(3):
        att_w,_ = model.calc_att_w(x_ids[i])
        print(i,"att")
        for a in att_w:
            print(a.data[0])
