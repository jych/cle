import ipdb
import numpy as np
import theano.tensor as T

from itertools import izip
from cle.cle.layers import StemCell, RandomCell, InitCell
from cle.cle.utils import tolist
from theano.compat.python2x import OrderedDict
from theano.tensor.nnet import conv2d, ConvOp
from theano.tensor.sensor.downsample import max_pool_2d


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



def conv2d(X, W, shape, border_mode):
    """
    Perform 2D convolution

    Parameters
    ----------
    X : Theano tensor4
        Inputs
    W : Theano sharedX
        Filters
    shape : tuple
        Shape of inputs
        (batch_size, num_channels, image_size, image_size)
    """
    z = conv2d(X, W,
        image_shape=shape,
        border_mode=border_mode

    )



