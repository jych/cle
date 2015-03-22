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
            T.tensordot(Fx, T.tensordot(x, Fy, [[2], [0]]), [[1], [1]])
        x_hat_ = gamma *\
            T.tensordot(Fx, T.tensordot(x_hat, Fy, [[2], [0]]), [[1], [1]])
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
            T.tensordot(Fy.T, T.tensordot(x, Fx, [[2], [0]]), [[1], [1]])
        return x_
