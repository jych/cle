import ipdb
import numpy as np
import theano.tensor as T

from itertools import izip
from cle.cle.layers import StemCell, RandomCell, InitCell
from cle.cle.utils import tolist, totuple
from theano.compat.python2x import OrderedDict
from theano.tensor.nnet import conv2d, ConvOp
from theano.tensor.sensor.downsample import max_pool_2d
    

# We need to write FF->Conv converter layer, and vice versa.
# We only need 2D->4D and 4D->2D since 3D
# (bs, ch, x) could be represented as (bs, ch, x, 1)
# and there is no case like 3D representation (timestep, batch, x)
# since rnn is handled by a giant scan configuration.
# Say giant scan because it could contain an MLP inside.
# Pooling layers exist separately in layers.py
# Batch normalization should also locate in layers.py
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
                 tiedbias=True,
                 stepsize=(1, 1),
                 bordermode='valid',
                 **kwargs):
        super(Conv2DLayer, self).__init__(**kwargs)
        self.nonlin = self.which_nonlin(unit)
        # filters are parent-wise-filterbank
        # type(filters) should be dict
        self.filters = filters
        # Shape should be (batch_size, num_channels, x, y)
        self.outshape = outshape
        self.tiedbias = tiedbias
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
        if self.tiedbias:
            z += self.params['b_'+self.name].dimshuffle('x', 0, 'x', 'x')
        else:
            z += self.params['b_'+self.name].dimshuffle('x', 0, 1, 2)
        z = self.nonlin(z)
        z.name = self.name
        return z

    def initialize(self):
        outshape = self.outshape
        for i, parent in enumerate(self.parent):
            pshape = parent.outshape
            self.alloc(self.init_W.get((parent.nout, self.nout),
                                       'W_'+parent.name+self.name))
        self.alloc(self.init_b.get(self.nout, 'b_'+self.name))

        
class ConverterLayer(StemCell):
    """
    Convert 2D matrix to 4D tensor
    or vice versa

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 targetshape=None,
                 axes=('b', 'c', 'x', 'y'),
                 **kwargs):
        super(ConverterLayer, self).__init__(**kwargs)
        self.targetshape = targetshape
        if len(targetshape) == 2 or targetshape=None:
            convert_type = 'convert2matrix'
        elif len(targetshape) == 4:
            convert_type = 'convert2tensor4'
        self.fprop = self.which_convert(convert_type)

    def which_init(self, which):
        return getattr(self, which)

    def convert2matrix(self, x):
        # Assume that axes of x is always ('b', 'c', 'x', 'y')
        refaxes = ('b', 'c', 'x', 'y')
        newaxes = ()
        for axis in self.axes:
            newaxes += (refaxes.index(axis))
        batch_size = x.shape[0]
        x = x.dimshuffle(newaxes)
        z = x.reshape((batch_size, -1))
        z.name = self.name
        return z

    def convert2tensor4(self, x):
        z = x.reshape(self.targetshape)
        z.name = self.name
        return z

    def initialize(self):
        pass
