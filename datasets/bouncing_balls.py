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
    def load(self, path):
        data = np.load(path)
        X = data[:, :-1, :]
        y = data[:, 1:, :]
        return (X, y)

    def theano_vars(self):
        return [T.ftensor3('x'), T.ftensor3('y')]
