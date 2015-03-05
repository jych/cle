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
    def __init__(self, name, path, batchsize=None):
        self.name = name
        self.path = path
        self.batchsize = batchsize
        self.data = self.load_data(path)
        self.nexp = self.num_examples()
        self.batchsize = self.nexp if batchsize is None else batchsize
        self.nbatch = int(np.float(self.nexp / float(self.batchsize)))
        self.index = -1

    def num_examples(self):
        return self.data[0].shape[0]

    def load_data(self, path):
        X = np.load(path[0])
        y = np.load(path[1])
        return (X, y)

    def theano_vars(self):
        return [T.fmatrix('x'), T.fmatrix('y')]
