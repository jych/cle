import ipdb
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
    def fprop(self, X):
        if len(X) != len(self.parent):
            raise AttributeError("The number of inputs doesn't match "
                                 "with the number of parents.")
        # X could be a list of inputs.
        # depending the number of parents.
        z = T.zeros((X[0].shape[0], self.nout))
        for x, (parname, parout) in izip(X, self.parent.items()):
            W = self.params['W_'+parname+'__'+self.name]
            z += T.dot(x[:, :parout], W)
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
        v_mean, v, h_mean, h = self.gibbs_step(X)
        return v, h

    def gibbs_step(self, X):
        x = X[0]
        parname, parout = self.parent.items()[0]
        W = self.params['W_'+parname+'__'+self.name]
        h_mean = T.dot(x[:, :parout]/X[3], W) + X[1]
        h_mean = T.nnet.sigmoid(h_mean)
        h = self.theano_rng.binomial(size=h_mean.shape, n=1, p=h_mean,
                                     dtype=theano.config.floatX)
        v_mean = T.dot(h, W.T) + X[2]
        epsilon = self.theano_rng.normal(size=v_mean.shape, avg=0.,
                                         std=1., dtype=theano.config.floatX)
        v = v_mean + X[3] * epsilon
        return v_mean, v, h_mean, h

    def free_energy(self, v, X):
        W = self.params['W_'+parname+'__'+self.name]
        bias_term = 0.5*(((X[2] - v)/X[3])**2).sum(axis=1) 
        hidden_term = T.log(1 + T.exp(T.dot(v/X[3], W) + X[1])).sum(axis=1)
        FE = bias_term -hidden_term
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
