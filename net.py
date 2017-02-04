# -*- coding: utf-8 -*-

import chainer.links as L
import chainer.functions as F

from chainer import Chain

class NStepLSTM(L.NStepLSTM):

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

class NstepLstmNet(Chain):

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
        self.__vec_size = filter_width
        super(SimpleCnnNet, self).__init__(
            embed = L.EmbedID(n_vocab, filter_width, ignore_label=-1),
            cnn1 = L.Convolution2D(1, output_channel, (filter_height, filter_width), use_cudnn=use_cudnn),
            bnorm1=L.BatchNormalization(output_channel),
            l1 = L.Linear(output_channel, n_unit),
            l2 = L.Linear(n_unit, n_out),
        )

    def __call__(self, x, train=True):
        h_x = self.embed(x)
        n_words = h_x.shape[1]
        h_x = F.expand_dims(h_x, 1)
        h_x = F.relu(self.bnorm1(self.cnn1(h_x)))
        h_x = F.max_pooling_2d(h_x, (n_words, self.__vec_size))
        h_x = F.relu(self.l1(h_x))
        return self.l2(h_x)
