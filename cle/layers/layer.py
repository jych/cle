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
                 ignoreborder=False,
                 **kwargs):
        super(MaxPool2D, self).__init__(**kwargs)
        parent = unpack(self.parent)
        # Shape should be (batch_size, num_channels, x, y)
        parshape = parent.outshape
        poolsize = totuple(poolsize)
        poolstride = totuple(poolstride)
        if ignoreborder:
            newx = (parshape[2] - poolsize[0]) // poolstride[0] + 1
            newy = (parshape[3] - poolsize[1]) // poolstride[1] + 1
        else:
            if poolstride[0] > poolsize[0]:
                newx = (parshape[2] - 1) // poolstride[0] + 1
            else:
                newx = max(0, (parshape[2] - 1 - poolsize[0]) //
                           poolstride[0] + 1) + 1
            if poolstride[1] > poolsize[1]:
                newy = (parshape[3] - 1) // poolstride[1] + 1
            else:
                newy = max(0, (parshape[3] - 1 - poolsize[1]) //
                           poolstride[1] + 1) + 1
        outshape = (parshape[0], parshape[1], newx, newy)
        self.ignoreborder = ignoreborder
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
