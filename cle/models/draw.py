import ipdb
import numpy as np

from cle.cle.layers import InitCell, StemCell
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.layers.recurrent import RecurrentLayer


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
                 width=0,
                 hegiht=0,
                 init_U=InitCell('randn'),
                 **kwargs):
        super(ReadLayer, self).__init__(init_U, **kwargs)
        self.N = N
        self.width = width
        self.height = height

    def fprop(self, XH):
        X, H = XH
        x = X[0]
        x_hat = X[1]
        z = T.zeros((self.batch_size, self.nout))
        for h, (recname, recout) in izip(H, self.recurrent.items()):
            U = self.params['U_'+recname+self.name]
            z += T.dot(h[:, :recout], U)
        z += self.params['b_'+self.name]
        centex = z[:, 0]
        centey = z[:, 1]
        logvar = z[:, 2]
        logdel = z[:, 3]
        loggam = z[:, 4]

        centx = (self.width + 1) * (centex + 1) / 2.
        centy = (self.height + 1) * (centey + 1) / 2.
        sigma = T.exp(0.5*logvar)
        gamma = T.exp(loggam).dimshuffle(0, 'x')
        delta = T.exp(logdel)
        delta = (max(self.width, self.height) - 1) * delta / (self.N - 1)

        Fx, Fy = self.filterbank(centx, centy, delta, sigma)
        x_ = gamma *\
            T.tensordot(Fx, T.tensordot(x, Fy, [[3], [0]]), [[1], [2]])
        x_hat_ = gamma *\
            T.tensordot(Fx, T.tensordot(x_hat, Fy, [[3], [0]]), [[1], [2]])
        return T.concatenate([x, x_hat], axis=2)

    def filter_bank(self, c_x, c_y, delta, sigma):
        tol = 1e-4
        mesh = T.arange(N) - (N/2) - 0.5

        a = T.arange(self.width).dimshuffle(0, 'x')
        b = T.arange(self.height).dimshuffle(0, 'x')
        mu_x = (c_x + delta).dimshuffle('x', 0) * mesh
        mu_y = (c_y + delta).dimshuffle('x', 0) * mesh

        #mu_x = (c_x + delta) * mesh
        #mu_y = (c_y + delta) * mesh
        #a = T.arange(self.width)
        #b = T.arange(self.height)
        #Fx = T.exp(-(a - mu_x.T)**2) / (2. * (sigma + tol)**2)
        #Fy = T.exp(-(b - mu_y.T)**2) / (2. * (sigma + tol)**2)
        Fx = T.exp(-(a - mu_x)**2) / (2. * (sigma + tol)**2)
        Fy = T.exp(-(b - mu_y)**2) / (2. * (sigma + tol)**2)
        Fx = Fx / Fx.sum(axis=0)
        Fy = Fy / Fy.sum(axis=0)
        return Fx, Fy

    def initialize(self):
        for recname, recout in self.recurrent.items():
            U_shape = (recout, 5)
            U_name = 'U_'+recname+self.name
            sel.alloc(self.init_U.get(U_shape, U_name))
        self.alloc(self.init_b.get(5, 'b_'+self.name))


class WriteLayer(ReadLayer):
    """
    Draw write layer

    Parameters
    ----------
    .. todo::
    """
    def fprop(self, XH):
        X, H = XH
        x = X[0]
        z = T.zeros((self.batch_size, self.nout))
        for x, (parname, parout) in izip(X, self.parent.items()):
            W = self.params['W_'+parname+self.name]
            z += T.dot(x[:, :parout], W)
        for h, (recname, recout) in izip(H, self.recurrent.items()):
            U = self.params['U_'+recname+self.name]
            z += T.dot(h[:, :recout], U)
        z += self.params['b_'+self.name]
        centex = z[:, 0]
        centey = z[:, 1]
        logvar = z[:, 2]
        logdel = z[:, 3]
        loggam = z[:, 4]

        centx = (self.width + 1) * (centex + 1) / 2.
        centy = (self.height + 1) * (centey + 1) / 2.
        sigma = T.exp(0.5*logvar)
        gamma = T.exp(loggam).dimshuffle(0, 'x')
        delta = T.exp(logdel)
        delta = (max(self.width, self.height) - 1) * delta / (self.N - 1)

        Fx, Fy = self.filterbank(centx, centy, delta, sigma)
        x_ = gamma *\
            T.tensordot(Fy.T, T.tensordot(x, Fx, [[3], [0]]), [[1], [2]])
        return x_

    def initialize(self):
        for parname, parout in self.parent.items():
            W_shape = (parout, self.nout)
            W_name = 'W_'+parname+self.name
            self.alloc(self.init_W.get(W_shape, W_name))
        for recname, recout in self.recurrent.items():
            U_shape = (recout, self.nout)
            U_name = 'U_'+recname+self.name
            sel.alloc(self.init_U.get(U_shape, U_name))
        self.alloc(self.init_b.get(self.nout, 'b_'+self.name))


class CanvasLayer(RecurrentLayer):
    """
    Canvas layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 is_write=0,
                 is_binary=0,
                 is_gaussian=0,
                 is_gaussian_mixture=0,
                 **kwargs):
        super(CanvasLayer, self).__init__(**kwargs)
        self.is_write = is_write
        if self.is_write:
            self.fprop = self.which_method('write')
        else:
            self.fprop = self.which_method('error')
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

    def write(self, XH):
        X, H = XH
        c_t = X[0]
        c_tm1 = H[0]
        z = c_tm1 + c_t
        z.name = self.name
        return z

    def error(self, XH):
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
        dic.pop('fprop')
        dic.pop('sample')
        return dic
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.is_write:
            self.fprop = self.which_method('write')
        else:
            self.fprop = self.which_method('error')
        if self.is_binary:
            self.sample = self.which_method('binary')
        elif self.is_gaussian:
            self.sample = self.which_method('gaussian')
        elif self.is_gmm:
            self.sample = self.which_method('gaussian_mixture')

    def initialize(self):
        pass
