import ipdb
import numpy as np
import theano.tensor as T

from itertools import izip
from cle.cle.layers import StemCell, RandomCell, InitCell
from cle.cle.utils import tolist
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
            z += T.dot(x[:, :parent.nout], self.params['W_'+parent.name+self.name])
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

