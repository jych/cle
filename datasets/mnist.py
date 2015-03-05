import ipdb
import numpy as np
import theano.tensor as T
from cle.cle.data import DesignMatrix


class MNIST(DesignMatrix):
    """
    MNIST batch provider

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
        data = np.load(path)
        if self.name == 'train':
            return data[0]
        elif self.name == 'valid':
            return data[1]
        elif self.name == 'test':
            return data[2]

    def theano_vars(self):
        return [T.fmatrix('x'), T.lvector('y')]
