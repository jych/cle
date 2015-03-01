import ipdb
import numpy as np
import theano.tensor as T

from itertools import izip
from cle.cle.layers import StemCell, RandomCell, InitCell
from cle.cle.utils import tolist, totuple, unpack
from theano.compat.python2x import OrderedDict
from theano.tensor.nnet import conv2d, ConvOp
from theano.tensor.signal.downsample import max_pool_2d
    

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
                 outshape,
                 tiedbias=True,
                 stepsize=(1, 1),
                 bordermode='valid',
                 **kwargs):
        super(Conv2DLayer, self).__init__(**kwargs)
        self.nonlin = self.which_nonlin(unit)
        # Shape should be (batch_size, num_channels, x, y)
        self.outshape = outshape
        self.tiedbias = tiedbias
        self.stepsize = totuple(stepsize)
        self.bordermode = bordermode

    def fprop(self, x):
        # Conv layer can have only one parent.
        # Later, we will extend to generalize this.
        # For now, we satisfy with fullyconnected layer
        # that can embed multiple conv layer parents
        # into same hidden space.
        x = unpack(x)
        parent = unpack(self.parent)
        z = T.zeros(self.outshape)
        W = self.params['W_'+parent.name+self.name]
        z += conv2d(x, W,
            image_shape=parent.outshape,
            subsample=self.stepsize,
            border_mode=self.bordermode,
            filter_shape=self.filtershape
        )
        if self.tiedbias:
            z += self.params['b_'+self.name].dimshuffle('x', 0, 'x', 'x')
        else:
            z += self.params['b_'+self.name].dimshuffle('x', 0, 1, 2)
        z = self.nonlin(z)
        z.name = self.name
        return z

    def initialize(self):
        parent = unpack(self.parent)
        outshape = self.outshape
        parshape = parent.outshape
        batchsize = outshape[0]
        nfilters = outshape[1]
        nchannels = parshape[1]
        if self.bordermode == 'valid':
            x = parshape[2] - outshape[2] + 1
            y = parshape[3] - outshape[3] + 1
        elif self.bordermode == 'full':
            x = outshape[2] - parshape[2] + 1
            y = outshape[3] - parshape[3] + 1
        W_shape = (nfilters, nchannels, x, y)
        self.filtershape = W_shape
        W_name = 'W_'+parent.name+self.name
        self.alloc(self.init_W.get(W_shape, W_name))
        b_name = 'b_'+self.name
        if self.tiedbias:
            b_shape = nchannels
            self.alloc(self.init_b.get(b_shape, b_name))
        else:
            b_shape = (nchannels, x, y)
            self.alloc(self.init_b.get(b_shape, b_name))


class ConvertLayer(StemCell):
    """
    Convert 2D matrix to 4D tensor
    or vice versa

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 outshape=None,
                 axes=('b', 'c', 'x', 'y'),
                 **kwargs):
        super(ConvertLayer, self).__init__(**kwargs)
        self.outshape = outshape
        if len(outshape) == 2 or outshape == None:
            convert_type = 'convert2matrix'
            self.nout = outshape[1]
        elif len(outshape) == 4:
            convert_type = 'convert2tensor4'
        self.fprop = self.which_convert(convert_type)
        self.axes = axes

    def which_convert(self, which):
        return getattr(self, which)

    def convert2matrix(self, x):
        x = unpack(x)
        # Assume that axes of x is always ('b', 'c', 'x', 'y')
        refaxes = ('b', 'c', 'x', 'y')
        newaxes = ()
        for axis in self.axes:
            newaxes += (refaxes.index(axis),)
        x = x.dimshuffle(newaxes)
        z = x.reshape((x.shape[0], -1))
        z.name = self.name
        return z

    def convert2tensor4(self, x):
        x = unpack(x)
        z = x.reshape(self.outshape)
        z.name = self.name
        return z

    def initialize(self):
        pass
