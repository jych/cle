import ipdb
import numpy as np
import theano.tensor as T

from cle.cle.layers import InitCell, StemCell
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.layers.recurrent import RecurrentLayer
from itertools import izip


class ReadLayer(RecurrentLayer):
    """
    Draw read layer

    Parameters
    ----------
    h_dec   : Linear
    x       : TensorVariable
    \hat{x} : Transformed TensorVariable
    """
    def __init__(self,
                 N,
                 img_shape=None,
                 **kwargs):
        super(ReadLayer, self).__init__(self_recurrent=0,
                                        **kwargs)
        self.N = N
        self.img_shape = img_shape

    def tensor_dot(self, X, Y):
        if Y.ndim == 4:
            Z = (X.dimshuffle(0, 1, 2, 'x') * Y).sum(axis=-2)
        else:
            Z = (X.dimshuffle(0, 1, 2, 'x') * Y.dimshuffle(0, 'x', 1, 2)).sum(axis=-2)
        return Z

    def fprop(self, XH):
        X, H = XH
        x = X[0]
        x_hat = X[1]
        z = T.zeros((self.batch_size, 5))
        for h, (recname, recout) in izip(H, self.recurrent.items()):
            U = self.params['U_'+recname+self.name]
            z += T.dot(h[:, :recout], U)
        z += self.params['b_'+self.name]
        batch_size, num_channel, width, height = self.img_shape
        x = x.reshape(self.img_shape)
        x_hat = x_hat.reshape(self.img_shape)

        centex = z[:, 0]
        centey = z[:, 1]
        logvar = z[:, 2]
        logdel = z[:, 3]
        loggam = z[:, 4]

        centx = (width + 1) * (centex + 1) / 2.
        centy = (height + 1) * (centey + 1) / 2.
        sigma = T.exp(0.5 * logvar)
        gamma = T.exp(loggam).dimshuffle(0, 'x', 'x', 'x')
        delta = T.exp(logdel)
        delta = (max(width, height) - 1) * delta / (self.N - 1)

        Fx, Fy = self.filter_bank(centx, centy, delta, sigma)
        x = (T.tensordot(Fy, T.tensordot(x, Fx.dimshuffle(0, 2, 1), [[3], [1]]).sum(axis=3), [[2], [2]]).sum(axis=2)).dimshuffle(0, 2, 1, 3)
        x_hat = (T.tensordot(Fy, T.tensordot(x_hat, Fx.dimshuffle(0, 2, 1), [[3], [1]]).sum(axis=3), [[2], [2]]).sum(axis=2)).dimshuffle(0, 2, 1, 3)
        x = x * gamma
        x_hat = x_hat * gamma
        reshape_shape = (batch_size, num_channel*self.N**2)
        return T.concatenate([x.reshape(reshape_shape), x_hat.reshape(reshape_shape)], axis=1)

    def filter_bank(self, c_x, c_y, delta, sigma):
        tol = 1e-4
        mesh = T.arange(self.N) - (0.5 * self.N) - 0.5

        a = T.arange(self.img_shape[3])
        b = T.arange(self.img_shape[2])
        mu_x = c_x.dimshuffle(0, 'x') + delta.dimshuffle(0, 'x') * mesh
        mu_y = c_y.dimshuffle(0, 'x') + delta.dimshuffle(0, 'x') * mesh

        Fx = T.exp(-(a - mu_x.dimshuffle(0, 1, 'x'))**2) / (2. * (sigma.dimshuffle(0, 'x', 'x') + tol)**2)
        Fy = T.exp(-(b - mu_y.dimshuffle(0, 1, 'x'))**2) / (2. * (sigma.dimshuffle(0, 'x', 'x') + tol)**2)

        Fx = Fx / Fx.sum(axis=-1).dimshuffle(0, 1, 'x')
        Fy = Fy / Fy.sum(axis=-1).dimshuffle(0, 1, 'x')
        return Fx, Fy

    def initialize(self):
        for recname, recout in self.recurrent.items():
            U_shape = (recout, 5)
            U_name = 'U_'+recname+self.name
            self.alloc(self.init_U.get(U_shape, U_name))
        self.alloc(self.init_b.get(5, 'b_'+self.name))


class WriteLayer(StemCell):
    """
    Draw write layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 N,
                 img_shape=None,
                 **kwargs):
        super(WriteLayer, self).__init__(**kwargs)
        self.N = N
        self.img_shape = img_shape

    def fprop(self, X):
        w, X = X[0], X[1:] 
        z = T.zeros((w.shape[0], 5))
        for x, (parname, parout) in izip(X, self.parent.items()):
            W = self.params['W_'+parname+self.name]
            z += T.dot(x[:, :parout], W)
        z += self.params['b_'+self.name]
        batch_size, num_channel, width, height = self.img_shape
        w = w.reshape(self.img_shape)
       
        centex = z[:, 0]
        centey = z[:, 1]
        logvar = z[:, 2]
        logdel = z[:, 3]
        loggam = z[:, 4]

        centx = (width + 1) * (centex + 1) / 2.
        centy = (height + 1) * (centey + 1) / 2.
        sigma = T.exp(0.5 * logvar)
        gamma = T.exp(loggam).dimshuffle(0, 'x', 'x', 'x')
        delta = T.exp(logdel)
        delta = (max(width, height) - 1) * delta / (self.N - 1)

        Fx, Fy = self.filter_bank(centx, centy, delta, sigma)
        w = (T.tensordot(Fy, T.tensordot(w, Fx.dimshuffle(0, 2, 1), [[3], [1]]).sum(axis=3), [[2], [2]]).sum(axis=2)).dimshuffle(0, 2, 1, 3)
        w = w * gamma 
        reshape_shape = (batch_size, num_channel*self.N**2)
        return w.reshape((reshape_shape))

    def filter_bank(self, c_x, c_y, delta, sigma):
        tol = 1e-4
        mesh = T.arange(self.N) - (0.5 * self.N) - 0.5

        a = T.arange(self.img_shape[3])
        b = T.arange(self.img_shape[2])
        mu_x = c_x.dimshuffle(0, 'x') + delta.dimshuffle(0, 'x') * mesh
        mu_y = c_y.dimshuffle(0, 'x') + delta.dimshuffle(0, 'x') * mesh

        Fx = T.exp(-(a - mu_x.dimshuffle(0, 1, 'x'))**2) / (2. * (sigma.dimshuffle(0, 'x', 'x') + tol)**2)
        Fy = T.exp(-(b - mu_y.dimshuffle(0, 1, 'x'))**2) / (2. * (sigma.dimshuffle(0, 'x', 'x') + tol)**2)

        Fx = Fx / Fx.sum(axis=-1).dimshuffle(0, 1, 'x')
        Fy = Fy / Fy.sum(axis=-1).dimshuffle(0, 1, 'x')
        return Fx, Fy

    def initialize(self):
        for parname, parout in self.parent.items():
            W_shape = (parout, 5)
            W_name = 'W_'+parname+self.name
            self.alloc(self.init_W.get(W_shape, W_name))
        self.alloc(self.init_b.get(5, 'b_'+self.name))


class CanvasLayer(RecurrentLayer):
    """
    Canvas layer

    Parameters
    ----------
    .. todo::
    """
    def fprop(self, XH):
        X, H = XH
        c_t = X[0]
        c_tm1 = H[0]
        z = c_tm1 + c_t
        z.name = self.name
        return z

    def initialize(self):
        pass


class ErrorLayer(RecurrentLayer):
    """
    Error layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 is_binary=0,
                 is_gaussian=0,
                 is_gaussian_mixture=0,
                 **kwargs):
        super(ErrorLayer, self).__init__(self_recurrent=0,
                                         **kwargs)
        self.is_binary = is_binary
        self.is_gaussian = is_gaussian
        self.is_gaussian_mixture = is_gaussian_mixture
        if self.is_binary:
            self.sample = self.which_method('binary')
        elif self.is_gaussian:
            self.sample = self.which_method('gaussian')
        elif self.is_gaussian_mixture:
            self.sample = self.which_method('gaussian_mixture')

    def which_method(self, which):
        return getattr(self, which)

    def fprop(self, XH):
        X, H = XH
        x = X[0]
        z = x - self.sample(H)
        z.name = self.name
        return z

    def binary(self, X):
        x = X[0]
        z = self.theano_rng.binomial(p=x, size=x.shape, dtype=x.dtype)
        return z

    def gaussian(self, X):
        mu = X[0]
        logvar = X[1]
        epsilon = self.theano_rng.normal(size=mu.shape,
                                         avg=0., std=1.,
                                         dtype=mu.dtype)
        z = mu + T.sqrt(T.exp(logvar)) * epsilon
        return z

    def gaussian_mixture(self, X):
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
            ),
            axis=1
        )
        mu = mu[T.arange(mu.shape[0]), :, idx]
        sig = T.sqrt(T.exp(logvar[T.arange(logvar.shape[0]), :, idx]))
        sample = self.theano_rng.normal(size=mu.shape,
                                        avg=mu, std=sig,
                                        dtype=mu.dtype)
        return sample

    def __getstate__(self):
        dic = self.__dict__.copy()
        dic.pop('sample')
        return dic
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.is_binary:
            self.sample = self.which_method('binary')
        elif self.is_gaussian:
            self.sample = self.which_method('gaussian')
        elif self.is_gmm:
            self.sample = self.which_method('gaussian_mixture')

    def initialize(self):
        pass       
