import ipdb
import copy
import functools
import numpy as np
import scipy
import theano.tensor as T

from theano.compat.python2x import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams
from cle.cle.cost import Gaussian, GMM, NllBin, NllMul, MSE
from cle.cle.utils import sharedX, tolist, unpack


class InitCell(object):
    """
    WRITEME

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 init_type='randn',
                 mean=0.,
                 stddev=0.01,
                 low=-0.08,
                 high=0.08,
                 **kwargs):
        super(InitCell, self).__init__(**kwargs)
        self.init_type = init_type
        if init_type is not None:
            self.init_param = self.which_init(init_type)
        self.mean = mean
        self.stddev = stddev
        self.low = low
        self.high = high

    def which_init(self, which):
        return getattr(self, which)

    def rand(self, shape):
        return np.random.uniform(self.low, self.high, shape)

    def randn(self, shape):
        return np.random.normal(self.mean, self.stddev, shape)

    def zeros(self, shape):
        return np.zeros(shape)

    def const(self, shape):
        return np.zeros(shape) + self.mean

    def ortho(self, shape):
        x = np.random.normal(self.mean, self.stddev, shape)
        return scipy.linalg.orth(x)

    def get(self, shape, name=None):
        return sharedX(self.init_param(shape), name)

    def setX(self, x, name=None):
        return sharedX(x, name)

    def __getstate__(self):
        dic = self.__dict__.copy()
        if self.init_type is not None:
            dic.pop('init_param')
        return dic
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.init_type is not None:
            self.init_param = self.which_init(self.init_type)


class NonlinCell(object):
    """
    WRITEME

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, unit=None):
        self.unit = unit
        if unit is not None:
            self.nonlin = self.which_nonlin(unit)
 
    def which_nonlin(self, which):
        return getattr(self, which)

    def linear(self, z):
        return z

    def relu(self, z):
        return z * (z > 0.)

    def sigmoid(self, z):
        return T.nnet.sigmoid(z)

    def softmax(self, z):
        return T.nnet.softmax(z)

    def tanh(self, z):
        return T.tanh(z)

    def steeper_sigmoid(self, z):
        return 1. / (1. + T.exp(-3.75 * z))

    def hard_tanh(self, z):
        return T.clip(z, -1., 1.)

    def hard_sigmoid(self, z):
        return T.clip(z + 0.5, 0., 1.)

    def __getstate__(self):
        dic = self.__dict__.copy()
        if self.unit is not None:
            dic.pop('nonlin')
        return dic
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.unit is not None:        
            self.nonlin = self.which_nonlin(self.unit)


class RandomCell(object):
    seed_rng = np.random.RandomState((2015, 2, 19))
    """
    WRITEME

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 theano_seed=None,
                 **kwargs):
        self.theano_seed = theano_seed

    def rng(self):
        if getattr(self, '_rng', None) is None:
            self._rng = np.random.RandomState(self.seed)
        return self._rng

    def seed(self):
        if getattr(self, '_seed', None) is None:
            self._seed = self.seed_rng.randint(np.iinfo(np.int32).max)
        return self._seed

    def theano_seed(self):
        if getattr(self, '_theano_seed', None) is None:
            self._theano_seed = self.seed_rng.randint(np.iinfo(np.int32).max)
        return self._theano_seed

    def theano_rng(self):
        if getattr(self, '_theano_rng', None) is None:
            self._theano_rng = MRG_RandomStreams(self.theano_seed)
        return self._theano_rng


class StemCell(NonlinCell):
    """
    WRITEME

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, parent, nout=None, init_W=InitCell('randn'),
                 init_b=InitCell('zeros'), name=None, **kwargs):
        super(StemCell, self).__init__(**kwargs)
        self.isroot = False
        if name is None:
            name = self.__class__.name__.lower()
        self.name = name
        self.nout = nout
        self.parent = tolist(parent)
        self.init_W = init_W
        self.init_b = init_b
        self.params = OrderedDict()

    def get_params(self):
        return self.params

    def fprop(self, x=None):
        raise NotImplementedError(
            str(type(self)) + " does not implement Layer.fprop.")

    def alloc(self, x):
        self.params[x.name] = x

    def initialize(self):
        for i, parent in enumerate(self.parent):
            W_shape = (parent.nout, self.nout)
            W_name = 'W_'+parent.name+self.name
            self.alloc(self.init_W.get(W_shape, W_name))
        self.alloc(self.init_b.get(self.nout, 'b_'+self.name))


class InputLayer(object):
    """
    Root layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, name, root, nout=None):
        self.isroot = True
        self.name = name
        root.name = self.name
        self.out = root
        self.nout = nout
        self.params = OrderedDict()

    def get_params(self):
        return self.params

    def initialize(self):
        pass


class OnehotLayer(StemCell):
    """
    Transform a scalar to one-hot vector

    Parameters
    ----------
    .. todo::
    """
    def fprop(self, x):
        x = unpack(x)
        z = T.zeros((x.shape[0], self.nout))
        z = T.set_subtensor(
            z[T.arange(x.size) % x.shape[0], x.T.flatten()], 1
        )
        z.name = self.name
        return z

    def initialize(self):
        pass


class MaskLayer(StemCell):
    """
    Masking layer

    Parameters
    ----------
    todo..
    """
    def fprop(self, xs):
        x = xs[0]
        t = xs[1]
        if x.ndim != 2:
            raise ValueError("Dimension of X should be 2,\
                              but got %d instead." % t.ndim)
        if t.ndim != 1:
            raise ValueError("Dimension of mask should be 1,\
                              but got %d instead." % t.ndim)
        #return x[t.nonzero()]
        return x * t[:, None]

    def initialize(self):
        pass


class CostLayer(StemCell):
    """
    Base cost layer

    Parameters
    ----------
    todo..
    """
    def __init__(self, use_sum=False, **kwargs):
        super(CostLayer, self).__init__(**kwargs)
        self.use_sum = use_sum
    
    def fprop(self, xs):
        raise NotImplementedError(
            str(type(self)) + " does not implement Layer.fprop.")

    def initialize(self):
        pass


class BinCrossEntropyLayer(CostLayer):
    """
    Binary cross-entropy layer

    Parameters
    ----------
    todo..
    """
    def fprop(self, xs):
        cost = NllBin(xs[0], xs[1])
        if self.use_sum:
            return cost.sum()
        else:
            return cost.mean()


class MulCrossEntropyLayer(CostLayer):
    """
    Multi cross-entropy layer

    Parameters
    ----------
    todo..
    """
    def fprop(self, xs):
        cost = NllMul(xs[0], xs[1])
        if self.use_sum:
            return cost.sum()
        else:
            return cost.mean()


class MSELayer(CostLayer):
    """
    Mean squared error layer

    Parameters
    ----------
    todo..
    """
    def fprop(self, xs):
        cost = MSE(xs[0], xs[1])
        if self.use_sum:
            return cost.sum()
        else:
            return cost.mean()


class GaussianLayer(CostLayer):
    """
    Linear Gaussian layer

    Parameters
    ----------
    todo..
    """
    def fprop(self, xs):
        if len(xs) != 3:
            raise ValueError("The number of inputs does not match.")
        cost = Gaussian(xs[0], xs[1], xs[2])
        if self.use_sum:
            return cost.sum()
        else:
            return cost.mean()


class GMMLayer(CostLayer):
    """
    Gaussian mixture model layer

    Parameters
    ----------
    todo..
    """
    def __init__(self,
                 ncoeff,
                 **kwargs):
        super(GMMLayer, self).__init__(**kwargs)
        if not isinstance(ncoeff, int):
            raise ValueError("Provide int number for this attribute.")
        else:
            if ncoeff < 2:
                raise ValueError("You want to have more than 2 Gaussians.")
        self.ncoeff = ncoeff

    def fprop(self, xs):
        if len(xs) != 4:
            raise ValueError("The number of inputs does not match.")
        cost = GMM(xs[0], xs[1], xs[2], xs[3])
        if self.use_sum:
            return cost.sum()
        else:
            return cost.mean()
