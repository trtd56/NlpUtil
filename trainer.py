# -*- coding: utf-8 -*-

import chainer.links as L
import chainer.functions as F
from chainer import optimizer, optimizers, training, iterators
from chainer.training import extensions
from chainer.datasets import tuple_dataset

class SoftMaxTrainer():

    def __init__(self, net):
        self.model = L.Classifier(net)

    def set_train_data(self, train_x, train_t, valid_x, valid_t, n_batch):
        train = tuple_dataset.TupleDataset(train_x, train_t)
        test  = tuple_dataset.TupleDataset(valid_x, valid_t)
        self.train_iter = iterators.SerialIterator(train, n_batch)
        self.test_iter  = iterators.SerialIterator(test, n_batch, repeat=False, shuffle=False)

    def set_trainer(self, out_dir, gpu, n_epoch, g_clip, opt_name, lr=None):
        if opt_name == "Adam":
            opt = getattr(optimizers, opt_name)()
        else:
            opt = getattr(optimizers, opt_name)(lr)
        opt.setup(self.model)
        opt.add_hook(optimizer.GradientClipping(g_clip))

        updater = training.StandardUpdater(self.train_iter, opt, device=gpu)
        self.trainer = training.Trainer(updater, (n_epoch, 'epoch'), out=out_dir)
        self.trainer.extend(extensions.Evaluator(self.test_iter, self.model, device=gpu))
        self.trainer.extend(extensions.dump_graph('main/loss'))
        self.trainer.extend(extensions.snapshot(), trigger=(n_epoch, 'epoch'))
        self.trainer.extend(extensions.LogReport())
        self.trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                                   'epoch', file_name='loss.png'))
        self.trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],
                                                   'epoch', file_name='accuracy.png'))
        self.trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss',
                                                    'main/accuracy', 'validation/main/accuracy',
                                                    'elapsed_time']))
        self.trainer.extend(extensions.ProgressBar())

    def start(self):
        self.trainer.run()

    def predict(self, x):
        pred = F.softmax(self.model.predictor(x, train=False))
        return pred.data
