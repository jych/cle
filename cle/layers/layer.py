import ipdb
import numpy as np
import theano.tensor as T

from itertools import izip

from cle.cle.layers import StemCell, RandomCell, InitCell
from cle.cle.util import tolist


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
        super(FullyConnectedLayer, self).__init__(**kwargs)
        self.nonlin = self.which_nonlin(unit)

    def fprop(self, xs):
        # xs could be a list of inputs.
        # depending the number of parents.
        z = T.zeros(self.nout)
        for x, parent in izip(xs, self.parent):
            z += T.dot(x, self.params['W_'+parent.name+self.name])
        z += self.params['b_'+self.name]
        z = self.nonlin(z)
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
    def __init__(self,
                 recurrent,
                 init_U=InitCell('ortho'),
                 **kwargs):
        super(SimpleRecurrent, self).__init__(**kwargs)
        self.recurrent = tolist(recurrent)
        self.init_U = init_U

    def get_init_state(self, batch_size):
        return T.zeros((batch_size, self.n_out))

    def initialize(self):
        super(SimpleRecurrent, self).initialize()
        for i, recurrent in enumerate(self.recurrent):
            self.alloc(self.init_U.get('U_'+recurrent.name+self.name,
                                       (recurrent.nout, self.nout)))


class SimpleRecurrent(RecurrentLayer):
    def __init__(self,
                 unit='tanh',
                 **kwargs):
        super(SimpleRecurrent, self).__init__(**kwargs)
        self.nonlin = self.which_nonlin(unit)

    def fprop(self, xh):
        # xs is a list of inputs
        xs, hs = xh
        z = T.zeros(self.nout)
        for x, parent in izip(xs, self.parent):
            z += T.dot(x, self.params['W_'+parent.name+self.name])
        for h, recurrent in izip(hs, self.recurrent):
            z += T.dot(h, self.params['U_'+recurrent.name+self.name])
        z += self.params['b_'+self.name]
        z = self.nonlin(z)
        z.name = self.name
        return z

    """
    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in :q
    """

"""
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
