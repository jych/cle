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
    def get_init_state(self, batch_size)
        n_out = self.get_dim(name)
        return T.zeros((batch_size, n_out))

class SimpleRecurrent(RecurrentLayer):
    def __init__(self,
                 context,
                 unit='tanh',
                 init_U=InitParams('ortho')
                 **kwargs):
        self.unit = unit
        super(SimpleRecurrent, self).__init__(**kwargs)
        self.init_U = init_U

    def fprop(self, h):
        x, z = h
        z_t = T.dot(x, self.W) + T.dot(z, self.U) + self.b
        z_t = self.nonlin(z_t)
        z_t.name = self.name
        return z_t

    def initialize(self):
        super(SimpleRecurrent, self).initialize() 
        for i, context in enumerate(self.context):
            self.allocate(self.init_U.get(self.name+'_U'+str(i+1),
                                          (context.nout, self.nout)))

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
