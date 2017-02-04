# -*- coding: utf-8 -*-

import numpy
from chainer import cuda, Variable
import chainer.functions as F

class XP:
    __lib = None

    @staticmethod
    def set_library(gpu=-1, seed=0):
        if gpu >= 0:
            # use GPU
            XP.__lib = cuda.cupy
            cuda.get_device(gpu).use()
            cuda.cupy.random.seed(seed)
        else:
            # use CPU
            XP.__lib = numpy
            numpy.random.seed(seed)

    @staticmethod
    def __zeros(shape, dtype):
        return Variable(XP.__lib.zeros(shape, dtype=dtype), volatile="auto")

    @staticmethod
    def fzeros(shape):
        return XP.__zeros(shape, XP.__lib.float32)

    @staticmethod
    def __array(array, dtype, is_volatile, transposition):
        volatile = "ON" if is_volatile else "OFF"
        if not transposition:
            return Variable(XP.__lib.array(array, dtype=dtype), volatile=volatile)
        else:
            return F.swapaxes(Variable(XP.__lib.array(array, dtype=dtype), volatile=volatile), 0, 1)

    @staticmethod
    def iarray(array, is_volatile=False, transposition=False):
        return XP.__array(array, XP.__lib.int32, is_volatile, transposition)

    @staticmethod
    def farray(array, is_volatile=False, transposition=False):
        return XP.__array(array, XP.__lib.float32, is_volatile, transposition)

    @staticmethod
    def to_cpu(data):
        if not XP.__lib == numpy:
            return cuda.to_cpu(data)
        else:
            return data

    @staticmethod
    def to_gpu(data):
        if not XP.__lib == numpy:
            return cuda.to_gpu(data)
        else:
            return data

    @staticmethod
    def lib():
        return XP.__lib
