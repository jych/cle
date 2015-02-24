import ipdb
import numpy as np
import theano.tensor as T

from cle.cle.layers import StemCell, RandomCell


class FullyConnectedLayer(StemCell):
    """
    Fully connected layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 unit,
                 **kwargs):
        self.unit = unit
        super(FullyConnectedLayer, self).__init__(**kwargs)

    def fprop(self, xx):
        # xx could be a list of inputs.
        # depending the number of parents.
        z = T.zeros(self.params[-1].shape)
        for i, x in enumerate(xx):
            z += T.dot(x, self.params[i])
        z += self.params[-1]
        z = self.nonlin(self.unit)(z)
        z.name = self.name
        return z


class ConvLayer(StemCell):
    """
    Convolutional layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self):
        raise NotImplementedError(
            str(type(self)) + " does not implement Layer.init.")

    def fprop(self, x=None):
        raise NotImplementedError(
            str(type(self)) + " does not implement Layer.fprop.")


class RecurrentLayer(StemCell):
    """
    Base recurrent layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self):
        raise NotImplementedError(
            str(type(self)) + " does not implement Layer.init.")

    def get_params(self):
        return []

    def fprop(self, x=None):
        raise NotImplementedError(
            str(type(self)) + " does not implement Layer.fprop.")

    """
    def get_init_state(self, batch_size)
        n_out = self.get_dim(name)
        return T.zeros((batch_size, n_out))
    """
"""
class SimpleRecurrent(RecurrentLayer):
    def __init__(self,
                 name,
                 n_in,
                 n_out,
                 unit='tanh',
                 init_W=InitParams('randn'),
                 init_U=InitParams('ortho'),
                 init_b=InitParams('zeros')):
        self.name = name
        self.nonlin = self.which_nonlin(unit)
        self.W = init_W.get(n_in, n_out)
        self.U = init_U.get(n_out, n_out)
        self.b = init_b.get(n_out)

    def get_params(self):
        return [self.W, self.U, self.b]

    def fprop(self, h):
        x, z = h
        z_t = T.dot(x, self.W) + T.dot(z, self.U) + self.b
        z_t = self.nonlin(z_t)
        z_t.name = self.name
        return z_t

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in :q


class LSTM(RecurrentLayer):
    def __init__(self):
        raise NotImplementedError(
            str(type(self)) + " does not implement Layer.init.")

    def get_params(self):
        return []

    def fprop(self, x=None):
        raise NotImplementedError(
            str(type(self)) + " does not implement Layer.fprop.")
"""
