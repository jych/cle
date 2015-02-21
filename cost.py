import numpy as np
import theano
import theano.tensor as T

from layer import *


def NllMul(y, y_hat):
    nll =  -T.sum(y * T.log(y_hat), axis=-1)
    return nll.mean()


def NllBin(y, y_hat):
    nll = -T.sum(y * T.log(y_hat) + (1-y) * T.log(1-y_hat), axis=-1)
    return nll.mean()


class CostLayer(Layer):
    """
    Implementations of cost layer

    Parameters
    ----------
    todo..
    """
    def __init__(self, name):
        pass

    def fprop(self, x, y):
        pass


class BinCrossEntropyLayer(CostLayer):
    """
    Implementations of cross-entropy

    Parameters
    ----------
    todo..
    """
    def __init__(self, name):
        self.name = name

    def fprop(self, y, y_hat):
        return NllBin(y, y_hat)


class MulCrossEntropyLayer(CostLayer):
    """
    Implementations of cross-entropy

    Parameters
    ----------
    todo..
    """
    def __init__(self, name):
        self.name = name

    def fprop(self, x):
        y_hat, y = x
        return NllMul(y, y_hat)
