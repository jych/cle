import ipdb
import numpy as np
import theano
import theano.tensor as T

from cle.cle.data import TemporalSeries


class BouncingBalls(TemporalSeries):
    """
    Bouncing balls batch provider

    Parameters
    ----------
    .. todo::
    """
    def load(self, path):
        data = np.load(path)
        N = len(data)
        train_N = int(N * 0.8)
        valid_N = int(N * 0.1)
        if self.name == 'train':
            data = data[:train_N]
        elif self.name == 'valid':
            data = data[train_N:train_N+valid_N]
        elif self.name == 'test':
            data = data[train_N+valid_N:]
        X = data[:, :-1, :]
        y = data[:, 1:, :]
        return (X, y)

    def theano_vars(self):
        return [T.tensor3('x', dtype=theano.config.floatX),
                T.tensor3('y', dtype=theano.config.floatX)]
