import ipdb
import numpy as np
import scipy
import theano
import theano.tensor as T

from cle.cle.utils import sharedX, tolist, unpack
from cle.cle.utils.gpu_op import softmax
from cle.cle.utils.op import add_noise

from theano.compat.python2x import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams


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
                 std_dev=0.01,
                 low=-0.08,
                 high=0.08,
                 **kwargs):
        super(InitCell, self).__init__(**kwargs)
        self.init_type = init_type
        if init_type is not None:
            self.init_param = self.which_init(init_type)
        self.mean = mean
        self.std_dev = std_dev
        self.low = low
        self.high = high

    def which_init(self, which):
        return getattr(self, which)

    def rand(self, shape):
        return np.random.uniform(self.low, self.high, shape)

    def randn(self, shape):
        return np.random.normal(self.mean, self.std_dev, shape)

    def zeros(self, shape):
        return np.zeros(shape)

    def ones(self, shape):
        return np.ones(shape)

    def const(self, shape):
        return np.zeros(shape) + self.mean

    def ortho(self, shape):
        x = np.random.normal(self.mean, self.std_dev, shape)
        return scipy.linalg.orth(x)

    def getX(self, shape, name=None):
        return sharedX(self.init_param(shape), name)

    def setX(self, x, name=None):
        return sharedX(x, name)

    def get(self, shape):
        return self.init_param(shape)

    def __getstate__(self):
        dic = self.__dict__.copy()
        if self.init_type is not None:
            dic.pop('init_param')
        return dic

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.init_type is not None:
            self.init_param = self.which_init(self.init_type)


class RandomCell(object):
    #seed_rng = np.random.RandomState((2015, 3, 24))
    seed_rng = np.random.RandomState(np.random.randint(1024))
    """
    WRITEME

    Parameters
    ----------
    .. todo::
    """
    def rng(self):
        if getattr(self, '_rng', None) is None:
            self._rng = np.random.RandomState(self.seed)
        return self._rng

    def seed(self):
        if getattr(self, '_seed', None) is None:
            self._seed = self.seed_rng.randint(np.iinfo(np.int32).max)
        return self._seed

    @property
    def theano_seed(self):
        if getattr(self, '_theano_seed', None) is None:
            self._theano_seed = self.seed_rng.randint(np.iinfo(np.int32).max)
        return self._theano_seed

    @theano_seed.setter
    def theano_seed(self, value):
        self._theano_seed = value

    @property
    def theano_rng(self):
        if getattr(self, '_theano_rng', None) is None:
            self._theano_rng = MRG_RandomStreams(self.theano_seed)
        return self._theano_rng

    @theano_rng.setter
    def theano_rng(self, theano_rng):
        self._theano_rng = theano_rng


class NonlinCell(RandomCell):
    """
    WRITEME

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, unit=None):
        self.unit = unit
        if unit is not None:
            self.nonlin = self.which_fn(unit)

    def which_fn(self, which):
        return getattr(self, which)

    def linear(self, z):
        return z

    def relu(self, z):
        return z * (z > 0.)

    def sigmoid(self, z):
        return T.nnet.sigmoid(z)

    def softmax(self, z):
        return T.nnet.softmax(z)

    def gpu_softmax(self, z):
        return softmax(z)

    def softplus(self, z):
        return T.nnet.softplus(z)

    def tanh(self, z):
        return T.tanh(z)

    def steeper_sigmoid(self, z):
        return 1. / (1. + T.exp(-3.75 * z))

    def hard_tanh(self, z):
        return T.clip(z, -1., 1.)

    def hard_sigmoid(self, z):
        return T.clip(z + 0.5, 0., 1.)

    def sigmoidal_spikenslab_relu(self, z):
        b = self.theano_rng.binomial(p=T.nnet.sigmoid(z - 3),
                                     size=z.shape,
                                     dtype=z.dtype)
        return z * b

    def gaussian_spikenslab_relu(self, z):
        z = T.exp(-T.sqr(z)) / float(np.sqrt(np.pi))
        b = self.theano_rng.binomial(p=(z.sum() - 3),
                                     size=z.shape,
                                     dtype=z.dtype)
        return z * b

    def __getstate__(self):
        dic = self.__dict__.copy()
        if self.unit is not None:
            dic.pop('nonlin')
        return dic

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.unit is not None:
            self.nonlin = self.which_fn(self.unit)


class StemCell(NonlinCell):
    """
    WRITEME

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 name,
                 parent=[],
                 parent_dim=[],
                 nout=None,
                 init_W=InitCell('randn'),
                 init_b=InitCell('zeros'),
                 cons=0.,
                 use_bias=1,
                 lr_scaler=1.,
                 x_as_index=0,
                 **kwargs):

        super(StemCell, self).__init__(**kwargs)

        if name is None:
            name = self.__class__.name__.lower()

        self.name = name
        self.nout = nout
        self.init_W = init_W
        self.init_b = init_b
        self.cons = cons
        self.x_as_index = x_as_index
        self.parent = OrderedDict()
        parent_dim = tolist(parent_dim)

        for i, par in enumerate(tolist(parent)):
            if len(parent_dim) != 0 and len(parent) != 0:
                if len(parent) != len(parent_dim):
                    raise AssertionError("You probably had a mistake providing,\
                                          write number of values. It will end,\
                                          up with a model containing a bug.")
                self.parent[par] = parent_dim[i]
            else:
                self.parent[par] = None
        self.lr_scaler = lr_scaler
        self.use_bias = use_bias

    def fprop(self):
        raise NotImplementedError(
            str(type(self)) + " does not implement Layer.fprop.")

    def initialize(self):

        params = OrderedDict()

        for parname, parout in self.parent.items():
            W_shape = (parout, self.nout)
            W_name = 'W_' + parname + '__' + self.name
            params[W_name] = self.init_W.get(W_shape)

        if self.use_bias:
            params['b_'+self.name] = self.init_b.get(self.nout)

        return params


class OnehotLayer(StemCell):
    """
    Transform a scalar to one-hot vector

    Parameters
    ----------
    .. todo::
    """
    def fprop(self, x):

        x = unpack(x)
        x = T.cast(x, 'int32')
        z = T.zeros((x.shape[0], self.nout))
        z = T.set_subtensor(
            z[T.arange(x.size) % x.shape[0], x.T.flatten()], 1
        )
        z.name = self.name

        return z

    def initialize(self):
        pass


class RealVectorLayer(StemCell):
    """
    Continuous vector

    Parameters
    ----------
    .. todo::
    """
    def fprop(self, tparams):

        z = tparams['b_'+self.name]
        z = self.nonlin(z) + self.cons

        if self.nout == 1:
            z = T.addbroadcast(z, 1)

        z.name = self.name

        return z

    def initialize(self):

        params = OrderedDict()
        params['b_'+self.name] = self.init_b.get(self.nout)

        return params
