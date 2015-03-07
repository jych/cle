import ipdb
import copy
import numpy as np
import scipy
import theano.tensor as T

from theano.compat.python2x import OrderedDict
from cle.cle.cost import Gaussian, GMM, NllBin, NllMul, MSE
from cle.cle.layers import RandomCell, StemCell
from cle.cle.utils import sharedX, tolist, unpack, predict


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
    todo..
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
    todo..
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
    todo..
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
    todo..
    """
    def fprop(self, X):
        if len(X) != 3:
            raise ValueError("The number of inputs does not match.")
        cost = Gaussian(X[0], X[1], X[2])
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
                 use_sample=False,
                 **kwargs):
        super(GMMLayer, self).__init__(**kwargs)
        self.use_sample = use_sample
        if use_sample:
            self.fprop = self.which_method('sample')
        else:
            self.fprop = self.which_method('cost')

    def which_method(self, which):
        return getattr(self, which)

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
        logvar = X[1]
        coeff = X[2]
        mu = mu.reshape((mu.shape[0],
                         mu.shape[1]/coeff.shape[-1],
                         coeff.shape[-1]))
        logvar = logvar.reshape((logvar.shape[0],
                                 logvar.shape[1]/coeff.shape[-1],
                                 coeff.shape[-1]))
        idx = predict(
            self.theano_rng.multinomial(
                pvals=coeff,
                dtype=coeff.dtype
            )
        )
        mu = mu[T.arange(mu.shape[0]), :, idx]
        sig = T.sqrt(T.exp(logvar[T.arange(mu.shape[0]), :, idx]))
        sample = self.theano_rng.normal(size=mu.shape,
                                        avg=mu, std=sig,
                                        dtype=mu.dtype)
        return sample

    def __getstate__(self):
        dic = self.__dict__.copy()
        dic.pop('fprop')
        return dic
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.use_sample:
            self.fprop = which_method('sample')
        else:
            self.fprop = which_method('cost')
