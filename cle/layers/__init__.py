import ipdb
import numpy as np
import scipy
import theano.tensor as T

from theano.compat.python2x import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams
from cle.cle.cost import NllBin, NllMul, MSE
from cle.cle.util import sharedX, tolist, unpack


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
        self.init_param = self.which_init(init_type)
        self.mean = mean
        self.stddev = stddev
        self.low = low
        self.high = high

    def which_init(self, which):
        return getattr(self, which)

    def rand(self, x):
        return np.random.uniform(self.low, self.high, x.shape)

    def randn(self, x):
        return np.random.normal(self.mean, self.stddev, x.shape)

    def zeros(self, x):
        return np.zeros(x.shape)

    def const(self, x):
        return np.zeros(x.shape) + self.mean

    def ortho(self, x):
        x = np.random.normal(self.mean, self.stddev, x.shape)
        return scipy.linalg.orth(x)

    def get(self, name, shape):
        return sharedX(self.init_param(np.zeros(shape)), name)


class NonlinCell(object):
    """
    WRITEME

    Parameters
    ----------
    .. todo::
    """
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
                 init_b=InitCell('zeros'), name=None):
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
            self.alloc(self.init_W.get('W_'+parent.name+self.name,
                                       (parent.nout, self.nout)))
        self.alloc(self.init_b.get('b_'+self.name, self.nout))


class InputLayer(object):
    """
    Root layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, name, root, nout):
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


class CostLayer(StemCell):
    """
    Base cost layer

    Parameters
    ----------
    todo..
    """
    def fprop(self, x=None):
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
    def fprop(self, xx):
        return NllBin(xx[0], xx[1])


class MulCrossEntropyLayer(CostLayer):
    """
    Multi cross-entropy layer

    Parameters
    ----------
    todo..
    """
    def fprop(self, xx):
        return NllMul(xx[0], xx[1])


class MSELayer(CostLayer):
    """
    Mean squared error layer

    Parameters
    ----------
    todo..
    """
    def fprop(self, xx):
        return MSE(xx[0], xx[1])
