# -*- coding: utf-8 -*-

from chainer import Chain, ChainList

class FastQrnnNet(ChainList):

    def __init__(self, in_size, out_size, n_units, n_layer)

        super(FastQrnnNet, self).__init__()
        if n_layer == 1:
            self.add_link(QRNNLayer(in_size, out_size))
        elif n_layer == 2:
            self.add_link(QRNNLayer(in_size, n_units))
            self.add_link(QRNNLayer(n_units, out_size))
        else:
            self.add_link(QRNNLayer(in_size, n_units))
            for _ in range(2, n_layer):
                self.add_link(QRNNLayer(n_units, n_units))
            self.add_link(QRNNLayer(n_units, out_size))

        self.layers  = [layer for layer in self.children()]

    def __call__(self, x, train):
        for layer in self.layers:
            x = F.dropout(layer(x), train=train)
        return x
