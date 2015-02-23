import ipdb
import numpy as np
import theano.tensor as T

from layer import *


def NllBin(y, y_hat):
    """
    Binary cross-entropy

    Parameters
    ----------
    todo..
    """
    nll = -T.sum(y * T.log(y_hat) + (1-y) * T.log(1-y_hat), axis=-1)
    return nll.mean()


def NllMul(y, y_hat):
    """
    Multi cross-entropy

    Parameters
    ----------
    todo..
    """
    nll =  -T.sum(y * T.log(y_hat), axis=-1)
    return nll.mean()


def MSE(y, y_hat):
    """
    Mean squared error

    Parameters
    ----------
    todo..
    """
    mse =  T.sum(T.sqr(y - y_hat), axis=-1)
    return mse.mean()


class BinCrossEntropyLayer(Layer):
    """
    Binary cross-entropy layer

    Parameters
    ----------
    todo..
    """
    def __init__(self, name):
        self.name = name

    def fprop(self, y, y_hat):
        return NllBin(y, y_hat)


class MulCrossEntropyLayer(Layer):
    """
    Multi cross-entropy layer

    Parameters
    ----------
    todo..
    """
    def __init__(self, name):
        self.name = name

    def fprop(self, x):
        x.sort(key=lambda x:x.name)
        y, y_hat = x
        return NllMul(y, y_hat)


class MSELayer(Layer):
    """
    Mean squared error layer

    Parameters
    ----------
    todo..
    """
    def __init__(self, name):
        self.name = name

    def fprop(self, x):
        x.sort(key=lambda x:x.name)
        y, y_hat = x
        ipdb.set_trace()
        return NllMul(y, y_hat)       
