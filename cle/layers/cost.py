import ipdb
import copy
import numpy as np
import scipy
import theano
import theano.tensor as T

from cle.cle.cost import Gaussian, GMM, NllBin, NllMul, MSE
from cle.cle.layers import RandomCell, StemCell
from cle.cle.utils import sharedX, tolist, unpack, predict

from theano.compat.python2x import OrderedDict


class CostLayer(StemCell):
    """
    Base cost layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, use_sum=False, **kwargs):
        super(CostLayer, self).__init__(**kwargs)
        self.use_sum = use_sum

    def fprop(self, X):
        raise NotImplementedError(
            str(type(self)) + " does not implement Layer.fprop.")

    def initialize(self):
        pass


class BinCrossEntropyLayer(CostLayer):
    """
    Binary cross-entropy layer

    Parameters
    ----------
    .. todo::
    """
    def fprop(self, X):
        cost = NllBin(X[0], X[1])
        if self.use_sum:
            return cost.sum()
        else:
            return cost.mean()


class MulCrossEntropyLayer(CostLayer):
    """
    Multi cross-entropy layer

    Parameters
    ----------
    .. todo::
    """
    def fprop(self, X):
        cost = NllMul(X[0], X[1])
        if self.use_sum:
            return cost.sum()
        else:
            return cost.mean()


class MSELayer(CostLayer):
    """
    Mean squared error layer

    Parameters
    ----------
    .. todo::
    """
    def fprop(self, X):
        cost = MSE(X[0], X[1])
        if self.use_sum:
            return cost.sum()
        else:
            return cost.mean()


class GaussianLayer(CostLayer):
    """
    Linear Gaussian layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 use_sample=False,
                 **kwargs):
        super(GaussianLayer, self).__init__(**kwargs)
        self.use_sample = use_sample
        if use_sample:
            self.fprop = self.which_fn('sample')
        else:
            self.fprop = self.which_fn('cost')

    def cost(self, X):
        if len(X) != 3:
            raise ValueError("The number of inputs does not match.")
        cost = Gaussian(X[0], X[1], X[2])
        if self.use_sum:
            return cost.sum()
        else:
            return cost.mean()

    def sample(self, X):
        mu = X[0]
        sig= X[1]
        epsilon = self.theano_rng.normal(size=mu.shape,
                                         avg=0., std=1.,
                                         dtype=mu.dtype)
        z = mu + sig * epsilon
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


class GMMLayer(GaussianLayer):
    """
    Gaussian mixture model layer

    Parameters
    ----------
    .. todo::
    """
    def cost(self, X):
        if len(X) != 4:
            raise ValueError("The number of inputs does not match.")
        cost = GMM(X[0], X[1], X[2], X[3])
        if self.use_sum:
            return cost.sum()
        else:
            return cost.mean()

    def sample(self, X):

        mu = X[0]
        sig = X[1]
        coeff = X[2]

        mu = mu.reshape((mu.shape[0],
                         mu.shape[1]/coeff.shape[-1],
                         coeff.shape[-1]))

        sig = sig.reshape((sig.shape[0],
                           sig.shape[1]/coeff.shape[-1],
                           coeff.shape[-1]))

        idx = predict(
            self.theano_rng.multinomial(
                pvals=coeff,
                dtype=coeff.dtype
            ),
            axis=1
        )

        mu = mu[T.arange(mu.shape[0]), :, idx]
        sig = sig[T.arange(sig.shape[0]), :, idx]

        epsilon = self.theano_rng.normal(size=mu.shape,
                                         avg=0., std=1.,
                                         dtype=mu.dtype)

        z = mu + sig * epsilon

        return z

    """
    def argmax_mean(self, X):

        mu = X[0]
        coeff = X[1]

        mu = mu.reshape((mu.shape[0],
                         mu.shape[1]/coeff.shape[-1],
                         coeff.shape[-1]))

        sig = sig.reshape((sig.shape[0],
                           sig.shape[1]/coeff.shape[-1],
                           coeff.shape[-1]))

        idx = predict(coeff)
        mu = mu[T.arange(mu.shape[0]), :, idx]
        sig = sig[T.arange(sig.shape[0]), :, idx]

        epsilon = self.theano_rng.normal(size=mu.shape,
                                         avg=0., std=1.,
                                         dtype=mu.dtype)

        z = mu + sig * epsilon

        return z, mu
    """
    def argmax_mean(self, X):

        mu = X[0]
        sig = X[1]
        coeff = X[2]

        mu = mu.reshape((mu.shape[0],
                         mu.shape[1]/coeff.shape[-1],
                         coeff.shape[-1]))

        sig = sig.reshape((sig.shape[0],
                           sig.shape[1]/coeff.shape[-1],
                           coeff.shape[-1]))

        idx = predict(coeff)
        mu = mu[T.arange(mu.shape[0]), :, idx]
        sig = sig[T.arange(sig.shape[0]), :, idx]

        epsilon = self.theano_rng.normal(size=mu.shape,
                                         avg=0., std=1.,
                                         dtype=mu.dtype)

        z = mu + sig * epsilon

        return z, mu

    def sample_mean(self, X):

        mu = X[0]
        sig = X[1]
        coeff = X[2]

        mu = mu.reshape((mu.shape[0],
                         mu.shape[1]/coeff.shape[-1],
                         coeff.shape[-1]))

        sig = sig.reshape((sig.shape[0],
                           sig.shape[1]/coeff.shape[-1],
                           coeff.shape[-1]))

        idx = predict(
            self.theano_rng.multinomial(
                pvals=coeff,
                dtype=coeff.dtype
            ),
            axis=1
        )

        mu = mu[T.arange(mu.shape[0]), :, idx]
        sig = sig[T.arange(sig.shape[0]), :, idx]

        epsilon = self.theano_rng.normal(size=mu.shape,
                                         avg=0., std=1.,
                                         dtype=mu.dtype)

        z = mu + sig * epsilon

        return z, mu


class LaplaceLayer(GaussianLayer):
    """
    Linear Laplace layer

    Parameters
    ----------
    .. todo::
    """
    def sample(self, X):
        mu = X[0]
        sig = X[1]
        u = self.theano_rng.uniform(size=mu.shape,
                                    low=0., high=1.,
                                    dtype=mu.dtype)
        v = self.theano_rng.uniform(size=mu.shape,
                                    low=0., high=1.,
                                    dtype=mu.dtype)
        epsilon = T.log(u) - T.log(v)
        z = mu + sig * epsilon
        return z
