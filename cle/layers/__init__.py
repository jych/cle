import ipdb
import numpy as np
import scipy
import theano.tensor as T

from theano.compat.python2x import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams
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


class RandomCell(object):
    seed_rng = np.random.RandomState((2015, 3, 24))
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
    def __init__(self, unit='linear'):
        self.unit = unit
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
        dic.pop('nonlin')
        return dic

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.nonlin = self.which_nonlin(self.unit)


class StemCell(NonlinCell):
    """
    WRITEME

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, parent=[], parent_dim=[], nout=None, init_W=InitCell('randn'),
                 init_b=InitCell('zeros'), cons=0., name=None, lr_scaler=None, **kwargs):
        super(StemCell, self).__init__(**kwargs)
        if name is None:
            name = self.__class__.name__.lower()
        self.name = name
        self.nout = nout
        self.init_W = init_W
        self.init_b = init_b
        self.cons = cons
        self.parent = OrderedDict()
        parent_dim = tolist(parent_dim)
        for i, par in enumerate(tolist(parent)):
            if len(parent_dim) != 0:
                self.parent[par] = parent_dim[i]
            else:
                self.parent[par] = None
        self.params = OrderedDict()
        self.lr_scaler = lr_scaler

    def get_params(self):
        return self.params

    def fprop(self, x=None):
        raise NotImplementedError(
            str(type(self)) + " does not implement Layer.fprop.")

    def alloc(self, x):
        self.params[x.name] = x

    def initialize(self):
        for parname, parout in self.parent.items():
            W_shape = (parout, self.nout)
            W_name = 'W_'+parname+'__'+self.name
            self.alloc(self.init_W.get(W_shape, W_name))
        self.alloc(self.init_b.get(self.nout, 'b_'+self.name))


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


class ProjectionLayer(StemCell):
    """
    Transform a scalar to one-hot vector

    Parameters
    ----------
    .. todo::
    """
    def fprop(self, x):
        x = unpack(x)
        z = T.zeros((x.shape[0], self.nout))
        for x, (parname, parout) in izip(X, self.parent.items()):
            W = self.params['W_'+parname+'__'+self.name]
            z += T.dot(x[:, :parout], W)
        z.name = self.name
        return z

    def initialize(self):
        for parname, parout in self.parent.items():
            W_shape = (parout, self.nout)
            W_name = 'W_'+parname+'__'+self.name
            self.alloc(self.init_W.get(W_shape, W_name))


class ConcLayer(StemCell):
    """
    Concatenate two tensor varaibles

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 axis=-1,
                 **kwargs):
        super(ConcLayer, self).__init__(**kwargs)
        self.axis = axis

    def fprop(self, X):
        x = X[0]
        y = X[1]
        z = T.concatenate([x[:, y.shape[-1]:], y], axis=self.axis)
        z.name = self.name
        return z

    def initialize(self):
        pass


class RealVectorLayer(StemCell):
    """
    Concatenate two tensor varaibles

    Parameters
    ----------
    .. todo::
    """
    def fprop(self, X=None):
        z = self.params['b_'+self.name]
        if z.ndim == 1:
            z = z.dimshuffle('x', 0)
        z = self.nonlin(z) + self.cons
        z.name = self.name
        return z

    def initialize(self):
        b_name = 'b_' + self.name
        self.alloc(self.init_b.get(self.nout, b_name))
        self.out = self.params[b_name]
