import ipdb
import numpy as np
import theano.tensor as T

from itertools import izip
from cle.cle.layers import StemCell, RandomCell, InitCell
from cle.cle.util import tolist
from theano.compat.python2x import OrderedDict


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
    Abstract class for recurrent layers

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 batch_size,
                 recurrent=[],
                 init_U=InitCell('ortho'),
                 **kwargs):
        super(RecurrentLayer, self).__init__(**kwargs)
        self.recurrent = tolist(recurrent)
        self.recurrent.append(self)
        self.batch_size = batch_size
        self.init_U = init_U
        self.init_states = OrderedDict()

    def get_init_state(self):
        state = T.zeros((self.batch_size, self.nout))
        state = T.unbroadcast(state, *range(self.dim))
        return state

    def initialize(self):
        super(RecurrentLayer, self).initialize()
        for i, recurrent in enumerate(self.recurrent):
            self.alloc(self.init_U.get('U_'+recurrent.name+self.name,
                                       (recurrent.nout, self.nout)))


class SimpleRecurrent(RecurrentLayer):
    """
    Vanilla recurrent layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 unit='tanh',
                 **kwargs):
        super(SimpleRecurrent, self).__init__(**kwargs)
        self.nonlin = self.which_nonlin(unit)

    def fprop(self, xh):
        # xh is a list of inputs: [state_belows, state_befores]
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


class LSTM(SimpleRecurrent):
    """
    Long short-term memory

    Parameters
    ----------
    .. todo::
    """
    def fprop(self, xh):
        # xh is a list of inputs: [state_belows, state_befores]
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
