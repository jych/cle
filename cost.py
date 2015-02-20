import numpy as np
import theano
import theano.tensor as T

def NLL_mul(probs, targets):
    return - T.sum(targets * T.log(probs)) / probs.shape[0]


def NLL_bin(probs, targets):
    return - T.sum(targets * T.log(probs) + (1-targets) * T.log(1-probs)) / probs.shape[0]



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


class MulticlassCostLayer(CostLayer):
    """
    Implementations of cost layer of multi-class case

    Parameters
    ----------
    todo..
    """
    def __init__(self, name, target):
        self.name = name
        self.target = target

    def fprop(self, p, y=None):
        y = self.target if y is None else y
        return - T.sum(y * T.log(p)) / p.shape[0]

