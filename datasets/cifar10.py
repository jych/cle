import ipdb
import numpy as np
import theano.tensor as T

from cle.cle.data import DesignMatrix


class CIFAR10(DesignMatrix):
    """
    CIFAR10 batch provider

    Parameters
    ----------
    .. todo::
    """
    def load(self, path):
        X = np.load(path[0])
        y = np.load(path[1])
        return (X, y)

    def theano_vars(self):
        return [T.fmatrix('x'), T.fmatrix('y')]
