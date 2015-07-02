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
        X = data[:, :-1, :]
        y = data[:, 1:, :]
        return (X, y)

    def theano_vars(self):
        return [T.tensor3('x', dtype=theano.config.floatX),
                T.tensor3('y', dtype=theano.config.floatX)]
