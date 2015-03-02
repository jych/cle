import ipdb
import numpy as np
import theano.tensor as T

from cle.cle.layers import StemCell, RandomCell, InitCell
from cle.cle.utils import tolist, totuple, unpack
from theano.compat.python2x import OrderedDict
from theano.tensor.signal.downsample import max_pool_2d

class MaxPool2D(StemCell):
    """
    2D Maxpooling layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 poolsize=(2, 2),
                 poolstride=(2, 2),
                 **kwargs):
        super(Conv2DLayer, self).__init__(**kwargs)
        # Shape should be (batch_size, num_channels, x, y)
        parent = unpack(parent)
        parshape = parent.outshape
        poolsize = totuple(poolsize)
        poolstride = totuple(poolstride)
        if (parshape[0] - poolsize[0]) % poolstride[0] != 0 or\
            (parshape[1] - poolsize[1]) % poolstride[1] != 0:
            raise ValueError("Detector layer shape should be
                              divisible, but remainder has detected")
        outshape[0] = (parshape[0] - poolsize[0]) / poolstride[0] + 1
        outshape[1] = (parshape[1] - poolsize[1]) / poolstride[1] + 1
        self.outshape = outshape
        self.poolsize = poolsize
        self.poolstride = poolstride

    def fprop(self, x):
        x = unpack(x)
        parent = unpack(self.parent)
        z = max_pool_2d(x, self.poolsize, st=self.poolstride)
        z.name = self.name
        return z

    def initialize(self):
        pass
