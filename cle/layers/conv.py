import ipdb
import numpy as np
import theano.tensor as T

from itertools import izip
from cle.cle.layers import StemCell, RandomCell, InitCell
from cle.cle.utils import tolist, totuple
from theano.compat.python2x import OrderedDict
from theano.tensor.nnet import conv2d, ConvOp
from theano.tensor.sensor.downsample import max_pool_2d
    


# We need to write FF->Conv converter
# and vice versa
# we only need 2D->4D and 4D->2D since 3D
# (bs, ch, x) could be represented as (bs, ch, x, 1)
# and there is no such case 3D representing (timestep, batch, x)
# since rnn is handled in giant scan configuration.
# I said giant because it could contain an MLP.
class Conv2DLayer(StemCell):
    """
    2D Convolutional layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 unit,
                 filters,
                 outshape,
                 stepsize=(1, 1),
                 bordermode='valid',
                 **kwargs):
        super(ConvLayer, self).__init__(**kwargs)
        self.nonlin = self.which_nonlin(unit)
        # filters are parent-wise-filterbank
        # type(filters) should be dict
        self.filters = filters
        self.outshape = outshape
        self.stepsize = totuple(stepsize)
        self.bordermode = bordermode

    def fprop(self, xs):
        # xs could be a list of inputs.
        # depending the number of parents.
        z = T.zeros(self.outshape)
        for x, parent in izip(xs, self.parent):
            w = self.params['W_'+parent.name+self.name]
            z += conv2d(
                inputs=x,
                filters=w,
                subsample=self.stepsize,
                image_shape=parent.outshape,
                border_mode=self.bordermode
            )
        z += self.params['b_'+self.name].dimshuffle('x', 0, 1, 2)
        z = self.nonlin(z)
        z.name = self.name
        return z
