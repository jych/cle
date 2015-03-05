import ipdb
import numpy as np
import theano.tensor as T
from cle.cle.data import DesignMatrix


class BouncingBalls(DesignMatrix):
    """
    Bouncing balls batch provider

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, **kwargs):
        super(BouncingBalls, self).__init__(**kwargs)
        self.index = -1

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
