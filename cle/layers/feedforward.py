import ipdb
import numpy as np
import theano
import theano.tensor as T

from cle.cle.layers import StemCell
from itertools import izip


class FullyConnectedLayer(StemCell):
    """
    Fully connected layer

    Parameters
    ----------
    .. todo::
    """
    def fprop(self, X, use_noisy_params=False, ndim=None):
        if len(X) != len(self.parent):
            raise AttributeError("The number of inputs doesn't match "
                                 "with the number of parents.")
        # X could be a list of inputs.
        # depending the number of parents.
        if ndim is None:
            ndim = 2
        elif type(ndim) is list:
            ndim = np.array([x.ndim for x in X]).max()
        if ndim == 2:
            z = T.zeros((X[0].shape[0], self.nout))
        if ndim == 3:
            z = T.zeros((X[0].shape[0], X[0].shape[1], self.nout))
        for x, (parname, parout) in izip(X, self.parent.items()):
            if use_noisy_params:
                W = self.noisy_params['W_'+parname+'__'+self.name]
            else:
                W = self.params['W_'+parname+'__'+self.name]
            if x.ndim == 1:
                if 'int' not in x.dtype:
                    x = T.cast(x, 'int64')
                if z.ndim == 2:
                    z += W[x]
                elif z.ndim == 3:
                    z += W[x][None, :, :]
            elif x.ndim == 2:
                if ndim == 2:
                    z += T.dot(x[:, :parout], W)
                if ndim == 3:
                    z += T.dot(x[:, :parout], W)[None, :, :]
            elif x.ndim == 3:
                if z.ndim != 3:
                    raise ValueError("your target ndim is less than source ndim")
                z += T.dot(x[:, :, :parout], W)
        z += self.params['b_'+self.name]
        z = self.nonlin(z) + self.cons
        z.name = self.name
        return z


class GRBM(StemCell):
    """
    Gaussian restrcited Boltzmann Machine

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 k_step=1,
                 **kwargs):
        super(GRBM, self).__init__(**kwargs)
        self.k_step = k_step

    def initialize(self):
        parname, parout = self.parent.items()[0]
        W_shape = (parout, self.nout)
        W_name = 'W_'+parname+'__'+self.name
        self.alloc(self.init_W.get(W_shape, W_name))

    def fprop(self, X):
        if len(X) != len(self.parent):
            raise AttributeError("The number of inputs doesn't match "
                                 "with the number of parents.")
        # X could be a list of inputs.
        # depending the number of parents.
        v = X[0]
        for i in xrange(self.k_step):
            v_mean, v, h_mean, h = self.gibbs_step(v, X[1], X[2], X[3])
        return v, h

    def gibbs_step(self, x, bh, bx, x_sig):
        parname, parout = self.parent.items()[0]
        W = self.params['W_'+parname+'__'+self.name]
        h_mean = T.nnet.sigmoid(T.dot(x[:, :parout]/(x_sig**2), W) + bh)
        h = self.theano_rng.binomial(size=h_mean.shape, n=1, p=h_mean,
                                     dtype=theano.config.floatX)
        v_mean = T.dot(h, W.T) + bx
        epsilon = self.theano_rng.normal(size=v_mean.shape, avg=0., std=1.,
                                         dtype=theano.config.floatX)
        v = v_mean + x_sig * epsilon
        return v_mean, v, h_mean, h

    def free_energy(self, v, X):
        W = self.params['W_'+parname+'__'+self.name]
        squared_term = 0.5 * ((X[2] - v) / X[3])**2
        hid_inp = T.dot(v/(X[3]**2), W) + X[1]
        FE = squared_term.sum(axis=1) - T.nnet.softplus(hid_inp).sum(axis=1)
        return FE

    def cost(self, X):
        v_mean, v, h_mean, h = self.gibbs_step(X)
        return (self.free_energy(X[0], X) - self.free_energy(v, X)).mean()

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
