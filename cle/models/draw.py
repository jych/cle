import ipdb
import numpy as np
import theano.tensor as T

from cle.cle.layers import InitCell, StemCell
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.layers.recurrent import RecurrentLayer
from itertools import izip


def batched_dot(A, B):     
    """Batched version of dot-product.     
       
    For A[dim_1, dim_2, dim_3] and B[dim_1, dim_3, dim_4] this         
    is \approx equal to:       
               
    for i in range(dim_1):     
        C[i] = tensor.dot(A, B)        
       
    Returns        
    -------        
        C : shape (dim_1 \times dim_2 \times dim_4)        
    """
    C = A.dimshuffle(0, 1, 2, 'x') * B.dimshuffle(0, 'x', 1, 2)      
    return C.sum(axis=-2)


class ReadLayer(StemCell):
    """
    Draw read layer

    Parameters
    ----------
    h_dec   : Linear
    x       : TensorVariable
    \hat{x} : Transformed TensorVariable
    """
    def __init__(self,
                 glimpse_shape=None,
                 input_shape=None,
                 **kwargs):
        super(ReadLayer, self).__init__(**kwargs)
        self.glimpse_shape = glimpse_shape
        self.input_shape = input_shape

    def fprop(self, X):
        x, x_hat, z, sig = X
        batch_size, num_channel, height, width = self.input_shape
        x = x.reshape((batch_size*num_channel, height, width))
        x_hat = x_hat.reshape((batch_size*num_channel, height, width))

        centey = z[:, 0]
        centex = z[:, 1]
        logdel = z[:, 2]
        loggam = z[:, 3]

        centy = 0.5 * (self.input_shape[2] + 1) * (centey + 1)
        centx = 0.5 * (self.input_shape[3] + 1) * (centex + 1)
        gamma = T.exp(loggam).dimshuffle(0, 'x')
        delta = T.exp(logdel)
        delta = (max(self.input_shape[2], self.input_shape[3]) - 1) * delta / (max(self.glimpse_shape[2], self.glimpse_shape[3]) - 1)

        Fy, Fx = self.filter_bank(centx, centy, delta, sig)
        x = batched_dot(batched_dot(Fy, x), Fx.transpose(0, 2, 1))
        x_hat = batched_dot(batched_dot(Fy, x_hat), Fx.transpose(0, 2, 1))
        reshape_shape = (batch_size,
                         num_channel*self.glimpse_shape[2]*self.glimpse_shape[3])
        return gamma * T.concatenate([x.reshape(reshape_shape), x_hat.reshape(reshape_shape)], axis=1)

    def filter_bank(self, c_x, c_y, delta, sigma):
        tol = 1e-4
        y_mesh = T.arange(self.glimpse_shape[2]) - 0.5 * self.glimpse_shape[2] - 0.5
        x_mesh = T.arange(self.glimpse_shape[3]) - 0.5 * self.glimpse_shape[3] - 0.5

        a = T.arange(self.input_shape[2])
        b = T.arange(self.input_shape[3])
        mu_y = c_y.dimshuffle(0, 'x') + delta.dimshuffle(0, 'x') * y_mesh
        mu_x = c_x.dimshuffle(0, 'x') + delta.dimshuffle(0, 'x') * x_mesh

        Fy = T.exp(-(a - mu_y.dimshuffle(0, 1, 'x'))**2) / 2. / sigma.dimshuffle(0, 'x', 'x')**2
        Fx = T.exp(-(b - mu_x.dimshuffle(0, 1, 'x'))**2) / 2. / sigma.dimshuffle(0, 'x', 'x')**2

        Fy = Fy / (Fy.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)
        Fx = Fx / (Fx.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)
        return Fy, Fx

    def initialize(self):
        pass


class WriteLayer(StemCell):
    """
    Draw write layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 glimpse_shape=None,
                 input_shape=None,
                 **kwargs):
        super(WriteLayer, self).__init__(**kwargs)
        self.glimpse_shape = glimpse_shape
        self.input_shape = input_shape

    def fprop(self, X):
        w, z, sig = X 
        batch_size, num_channel, height, width = self.glimpse_shape
        w = w.reshape((batch_size*num_channel, height, width))
       
        centey = z[:, 0]
        centex = z[:, 1]
        logdel = z[:, 2]
        loggam = z[:, 3]

        centy = 0.5 * (self.input_shape[2] + 1) * (centey + 1)
        centx = 0.5 * (self.input_shape[3] + 1) * (centex + 1)
        gamma = T.exp(loggam).dimshuffle(0, 'x')
        delta = T.exp(logdel)
        delta = (max(self.input_shape[2], self.input_shape[3]) - 1) * delta / (max(self.glimpse_shape[2], self.glimpse_shape[3]) - 1)

        Fy, Fx = self.filter_bank(centx, centy, delta, sig)
        I = batched_dot(batched_dot(Fy.transpose(0, 2, 1), w), Fx)
        reshape_shape = (batch_size, num_channel*self.input_shape[2]*self.input_shape[3])
        return I.reshape((reshape_shape)) / gamma

    def filter_bank(self, c_x, c_y, delta, sigma):
        tol = 1e-4
        y_mesh = T.arange(self.glimpse_shape[2]) - 0.5 * self.glimpse_shape[2] - 0.5
        x_mesh = T.arange(self.glimpse_shape[3]) - 0.5 * self.glimpse_shape[3] - 0.5

        a = T.arange(self.input_shape[2])
        b = T.arange(self.input_shape[3])
        mu_y = c_y.dimshuffle(0, 'x') + delta.dimshuffle(0, 'x') * y_mesh
        mu_x = c_x.dimshuffle(0, 'x') + delta.dimshuffle(0, 'x') * x_mesh

        Fy = T.exp(-(a - mu_y.dimshuffle(0, 1, 'x'))**2) / 2. / sigma.dimshuffle(0, 'x', 'x')**2
        Fx = T.exp(-(b - mu_x.dimshuffle(0, 1, 'x'))**2) / 2. / sigma.dimshuffle(0, 'x', 'x')**2

        Fy = Fy / (Fy.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)
        Fx = Fx / (Fx.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)
        return Fy, Fx

    def initialize(self):
        pass


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
            self.dist = self.which_method('binary')
        elif self.is_gaussian:
            self.dist = self.which_method('gaussian')
        elif self.is_gaussian_mixture:
            self.dist = self.which_method('gaussian_mixture')

    def which_method(self, which):
        return getattr(self, which)

    def fprop(self, XH):
        X, H = XH
        x = X[0]
        z = x - self.dist(H)
        z.name = self.name
        return z

    def binary(self, X):
        x = X[0]
        z = T.nnet.sigmoid(x)
        return z

    def gaussian(self, X):
        mu = X[0]
        sig = X[1]
        epsilon = self.theano_rng.normal(size=mu.shape,
                                         avg=0., std=1.,
                                         dtype=mu.dtype)
        z = mu + sig * epsilon
        return z

    def gaussian_mixture(self, X):
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
        sample = self.theano_rng.normal(size=mu.shape,
                                        avg=mu, std=sig,
                                        dtype=mu.dtype)
        return sample

    def __getstate__(self):
        dic = self.__dict__.copy()
        dic.pop('dist')
        return dic
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.is_binary:
            self.dist = self.which_method('binary')
        elif self.is_gaussian:
            self.dist = self.which_method('gaussian')
        elif self.is_gmm:
            self.dist = self.which_method('gaussian_mixture')

    def initialize(self):
        pass       
