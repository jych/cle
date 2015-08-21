import ipdb
import theano
import theano.tensor as T

from cle.cle.cost import (
    KLGaussianStdGaussian,
    KLGaussianGaussian
)
from cle.cle.layers import StemCell
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.utils import totuple, unpack
from cle.cle.utils.op import dropout

from theano.tensor.signal.downsample import max_pool_2d


class MaxPool2D(StemCell):
    """
    2D Maxpooling layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 pool_size=(2, 2),
                 pool_stride=(2, 2),
                 ignore_border=False,
                 set_shape=1,
                 **kwargs):
        super(MaxPool2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.ignore_border = ignore_border
        self.set_shape = set_shape
        if self.set_shape:
            self.initialize = self.which_fn('initialize_set_shape')
        else:
            self.initialize = self.which_fn('initialize_default')

    def initialize_set_shape(self):
        parname, parshape = unpack(self.parent.items())
        # Shape should be (batch_size, num_channels, x, y)
        pool_size = totuple(self.pool_size)
        pool_stride = totuple(self.pool_stride)
        if self.ignore_border:
            newx = (parshape[2] - pool_size[0]) // pool_stride[0] + 1
            newy = (parshape[3] - pool_size[1]) // pool_stride[1] + 1
        else:
            if pool_stride[0] > pool_size[0]:
                newx = (parshape[2] - 1) // pool_stride[0] + 1
            else:
                newx = max(0, (parshape[2] - 1 - pool_size[0]) //
                           pool_stride[0] + 1) + 1
            if pool_stride[1] > pool_size[1]:
                newy = (parshape[3] - 1) // pool_stride[1] + 1
            else:
                newy = max(0, (parshape[3] - 1 - pool_size[1]) //
                           pool_stride[1] + 1) + 1
        outshape = (parshape[0], parshape[1], newx, newy)
        self.outshape = outshape

    def fprop(self, x):
        x = unpack(x)
        z = max_pool_2d(x, self.pool_size, st=self.pool_stride)
        z.name = self.name
        return z

    def initialize_default(self):
        pass

    def __getstate__(self):
        dic = self.__dict__.copy()
        dic.pop('initialize')
        return dic

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.set_shape:
            self.initialize = self.which_fn('initialize_set_shape')
        else:
            self.initialize = self.which_fn('initialize_default')


class ClockworkLayer(StemCell):
    """
    Clockwork layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 N=1,
                 **kwargs):
        super(ClockworkLayer, self).__init__(**kwargs)
        self.N = N

    def fprop(self, z):
        z = theano.ifelse.ifelse(T.mod(idx, self.N) != 0,
                                 T.zeros_like(z),
                                 z)
        z.name = self.name
        return z


class PriorLayer(StemCell):
    """
    Prior layer which either computes
    the kl of VAE or generates samples using
    normal distribution when mod(t, N)==0

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 use_sample=False,
                 num_sample=1,
                 keep_dims=0,
                 **kwargs):
        super(PriorLayer, self).__init__(**kwargs)
        self.use_sample = use_sample
        self.keep_dims = keep_dims
        if self.use_sample:
            self.fprop = self.which_fn('sample')
        else:
            self.fprop = self.which_fn('cost')
        if use_sample:
            if num_sample is None:
                raise ValueError("If you are going to use sampling,\
                                  provide the number of samples.")
        self.num_sample = num_sample

    def cost(self, X):
        if len(X) != 2 and len(X) != 4:
            raise ValueError("The number of inputs does not match.")
        if len(X) == 2:
            return KLGaussianStdGaussian(X[0], X[1])
        elif len(X) == 4:
            if self.keep_dims:
                return KLGaussianGaussian(X[0], X[1], X[2], X[3], 1)
            else:
                return KLGaussianGaussian(X[0], X[1], X[2], X[3])

    def sample(self, X, num_sample=None):
        if len(X) != 2:
            raise ValueError("The number of inputs does not match.")
        if num_sample is None:
            num_sample = self.num_sample
        mu = X[0]
        sig = X[1]
        mu = mu.dimshuffle(0, 'x', 1)
        sig = sig.dimshuffle(0, 'x', 1)
        epsilon = self.theano_rng.normal(size=(mu.shape[0],
                                               num_sample,
                                               mu.shape[-1]),
                                         avg=0., std=1.,
                                         dtype=mu.dtype)
        z = mu + sig * epsilon
        z = z.reshape((z.shape[0] * z.shape[1], -1))
        return z

    def __getstate__(self):
        dic = self.__dict__.copy()
        dic.pop('fprop')
        return dic

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.use_sample:
            self.fprop = self.which_fn('sample')
        else:
            self.fprop = self.which_fn('cost')

    def initialize(self):
        pass
