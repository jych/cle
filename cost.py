import numpy as np
import theano
import theano.tensor as T

from layer import *


def NLLMul(y_hat, y):
    nll =  -T.sum(y * T.log(y_hat), axis=-1)
    return nll.mean()


def NLLBin(y_hat, y):
    nll = -T.sum(y * T.log(y_hat) + (1-y) * T.log(1-y_hat), axis=-1)
    return nll.mean()


#class CostLayer(Layer):
#    """
#    Implementations of cost layer

#    Parameters
#    ----------
#    todo..
#    """
#    def __init__(self, name):
#        pass

#    def fprop(self, x, y):
#        pass


#class MultiClassCostLayer(CostLayer):
#    """
#    Implementations of cost layer of multi-class case

#    Parameters
#    ----------
#    todo..
#    """
#    def __init__(self, name):
#        self.name = name

#    def fprop(self, p, y):
#        y = target if y is None else y
#        return - T.sum(y * T.log(p)) / p.shape[0]

