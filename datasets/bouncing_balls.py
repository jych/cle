import ipdb
import numpy as np
import theano.tensor as T
from cle.cle.data import TemporalSeries


class BouncingBalls(TemporalSeries):
    """
    Bouncing balls batch provider

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
        X = data[:, :-1, :]
        y = data[:, 1:, :]
        return (X, y)

    def next(self):
        self.index += 1
        if self.index < self.nbatch:
            return (self.batch(data, self.index) for data in self.data)
        else:
            self.index = -1
            raise StopIteration()

    def theano_vars(self):
        return [T.ftensor3('x'), T.ftensor3('y')]
