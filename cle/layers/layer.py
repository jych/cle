import ipdb
import theano

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
                 poolsize=(2, 2),
                 poolstride=(2, 2),
                 ignoreborder=False,
                 **kwargs):
        super(MaxPool2D, self).__init__(**kwargs)
        self.poolsize = poolsize
        self.poolstride = poolstride
        self.ignoreborder = ignoreborder

    def set_shape(self):
        parname, parshape = unpack(self.parent.items())
        # Shape should be (batch_size, num_channels, x, y)
        poolsize = totuple(self.poolsize)
        poolstride = totuple(self.poolstride)
        if self.ignoreborder:
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
        self.outshape = outshape

    def fprop(self, x):
        x = unpack(x)
        z = max_pool_2d(x, self.poolsize, st=self.poolstride)
        z.name = self.name
        return z

    def initialize(self):
        self.set_shape()
        pass


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


class DropoutLayer(StemCell):
    """
    Dropout layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 p=0.5,
                 train_scale=2,
                 test_scale=1,
                 is_test=0,
                 **kwargs):
        super(DropoutLayer, self).__init__(**kwargs)
        self.p = p
        self.train_scale = train_scale
        self.test_scale = test_scale
        self.is_test = is_test
        if self.is_test:
            self.fprop = self.which_prop('test_prop')
        else:
            self.fprop = self.which_prop('train_prop')

    def which_prop(self, which):
        return getattr(self, which)
    
    def train_prop(self, z):
        z = unpack(z)
        z = dropout(z, self.p, self.theano_rng)
        z *= self.train_scale
        z.name = self.name
        return z       

    def test_prop(self, z):
        z = unpack(z)
        z *= self.test_scale
        z.name = self.name
        return z

    def __getstate__(self):
        dic = self.__dict__.copy()
        dic.pop('fprop')
        return dic
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.is_test:
            
            self.fprop = self.which_prop('test_prop')
        else:
            self.fprop = self.which_prop('train_prop')
  
    def initialize(self):
        pass
