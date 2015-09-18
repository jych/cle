import ipdb
import numpy as np
import theano
import theano.tensor as T

from cle.cle.cost import (
    KLGaussianStdGaussian,
    KLGaussianGaussian
)
from cle.cle.layers import InitCell, StemCell
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.layers.recurrent import RecurrentLayer
from cle.cle.utils import totuple, unpack, sharedX
from cle.cle.utils.op import dropout

from itertools import izip

from theano.compat.python2x import OrderedDict
from theano.tensor.signal.downsample import max_pool_2d


class MaxPool2D(StemCell):
    """
    2D Maxpooling layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 pool_size=(2, 2),
                 pool_stride=(2, 2),
                 ignore_border=False,
                 set_shape=1,
                 **kwargs):
        super(MaxPool2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.ignore_border = ignore_border
        self.set_shape = set_shape

        if self.set_shape:
            self.initialize = self.which_fn('initialize_set_shape')
        else:
            self.initialize = self.which_fn('initialize_default')

    def initialize_set_shape(self):

        parname, parshape = unpack(self.parent.items())

        # Shape should be (batch_size, num_channels, x, y)
        pool_size = totuple(self.pool_size)
        pool_stride = totuple(self.pool_stride)

        if self.ignore_border:
            newx = (parshape[2] - pool_size[0]) // pool_stride[0] + 1
            newy = (parshape[3] - pool_size[1]) // pool_stride[1] + 1
        else:
            if pool_stride[0] > pool_size[0]:
                newx = (parshape[2] - 1) // pool_stride[0] + 1
            else:
                newx = max(0, (parshape[2] - 1 - pool_size[0]) //
                           pool_stride[0] + 1) + 1

            if pool_stride[1] > pool_size[1]:
                newy = (parshape[3] - 1) // pool_stride[1] + 1
            else:
                newy = max(0, (parshape[3] - 1 - pool_size[1]) //
                           pool_stride[1] + 1) + 1

        outshape = (parshape[0], parshape[1], newx, newy)
        self.outshape = outshape

    def fprop(self, x):

        x = unpack(x)
        z = max_pool_2d(x, self.pool_size, st=self.pool_stride)
        z.name = self.name

        return z

    def initialize_default(self):
        pass

    def __getstate__(self):
        dic = self.__dict__.copy()
        dic.pop('initialize')

        return dic

    def __setstate__(self, state):
        self.__dict__.update(state)

        if self.set_shape:
            self.initialize = self.which_fn('initialize_set_shape')
        else:
            self.initialize = self.which_fn('initialize_default')


class ClockworkLayer(StemCell):
    """
    Clockwork layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 N=1,
                 **kwargs):
        super(ClockworkLayer, self).__init__(**kwargs)
        self.N = N

    def fprop(self, z):
        z = theano.ifelse.ifelse(T.mod(idx, self.N) != 0,
                                 T.zeros_like(z),
                                 z)
        z.name = self.name
        return z


class PriorLayer(StemCell):
    """
    Prior layer which either computes
    the kl of VAE or generates samples using
    normal distribution when mod(t, N)==0

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 use_sample=False,
                 num_sample=1,
                 keep_dims=0,
                 **kwargs):
        super(PriorLayer, self).__init__(**kwargs)
        self.use_sample = use_sample
        self.keep_dims = keep_dims

        if self.use_sample:
            self.fprop = self.which_fn('sample')
        else:
            self.fprop = self.which_fn('cost')

        if use_sample:
            if num_sample is None:
                raise ValueError("If you are going to use sampling,\
                                  provide the number of samples.")
        self.num_sample = num_sample

    def cost(self, X):

        if len(X) != 2 and len(X) != 4:
            raise ValueError("The number of inputs does not match.")
        if len(X) == 2:
            return KLGaussianStdGaussian(X[0], X[1])
        elif len(X) == 4:
            if self.keep_dims:
                return KLGaussianGaussian(X[0], X[1], X[2], X[3], 1)
            else:
                return KLGaussianGaussian(X[0], X[1], X[2], X[3])

    def sample(self, X, num_sample=None):

        if len(X) != 2:
            raise ValueError("The number of inputs does not match.")

        if num_sample is None:
            num_sample = self.num_sample

        mu = X[0]
        sig = X[1]
        mu = mu.dimshuffle(0, 'x', 1)
        sig = sig.dimshuffle(0, 'x', 1)
        epsilon = self.theano_rng.normal(size=(mu.shape[0],
                                               num_sample,
                                               mu.shape[-1]),
                                         avg=0., std=1.,
                                         dtype=mu.dtype)
        z = mu + sig * epsilon
        z = z.reshape((z.shape[0] * z.shape[1], -1))

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

    def initialize(self):
        pass


class BatchNormLayer(StemCell):
    """
    Batch normalization layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, rho=0.5, eps=1e-6, **kwargs):
        super(BatchNormLayer, self).__init__(**kwargs)
        self.rho = rho
        self.eps = eps
        self.mu = sharedX(InitCell('zeros').get(self.nout), name='mu_'+self.name)
        self.sigma = sharedX(InitCell('ones').get(self.nout), name='sigma_'+self.name)

    def fprop(self, X, tparams, test=0, running_average=1, ndim=None):

        if len(X) != len(self.parent):
            raise AttributeError("The number of inputs does not match "
                                 "with the number of parents.")

        # X could be a list of inputs.
        # depending the number of parents.
        if ndim is None:
            ndims = [x.ndim for x in X]
            idx = np.argmax(ndims)
            ndim = np.maximum(np.array(ndims).max(), 2)

        z_shape = [X[idx].shape[i] for i in xrange(ndim-1)] + [self.nout]
        z = T.zeros(z_shape, dtype=theano.config.floatX)

        for x, (parname, parout) in izip(X, self.parent.items()):
            W = tparams['W_'+parname+'__'+self.name]

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
                    raise ValueError("your target ndim is less than the source ndim")
                z += T.dot(x[:, :, :parout], W)

        if not hasattr(self, 'use_bias'):
            z += tparams['b_'+self.name]
        elif self.use_bias:
            z += tparams['b_'+self.name]

        if ndim == 2:
            if not test:
                z_true = T.cast(T.neq(z.sum(axis=1), 0.0).sum(), dtype=theano.config.floatX)
                z_mu = z.sum(axis=0) / z_true
                z_sigma = T.sqrt(((z - z_mu[None, :])**2).sum(axis=0) / z_true)
                if running_average:
                    running_mu = theano.clone(self.mu, share_inputs=False)
                    running_mu.default_update = (self.rho * self.mu + (1 - self.rho) * z_mu)
                    running_sigma = theano.clone(self.sigma, share_inputs=False)
                    running_sigma.default_update = (self.rho * self.sigma + (1 - self.rho) * z_sigma)
                    z_mu += 0 * running_mu
                    z_sigma += 0 * running_sigma
                else:
                    mu = theano.clone(self.mu, share_inputs=False)
                    mu.default_update = z_mu
                    sigma = theano.clone(self.sigma, share_inputs=False)
                    sigma.default_update = z_sigma
                    z_mu += 0 * mu
                    z_sigma += 0 * sigma
            else:
                z_mu = self.mu
                z_sigma = self.sigma
            z_sigma += self.eps
            z = (z - z_mu[None, :]) * (tparams['gamma_'+self.name] / z_sigma)[None, :] + tparams['beta_'+self.name][None, :]
        if ndim == 3:
            if not test:
                z_true = T.cast(T.neq(z.sum(axis=2), 0.0).sum(), dtype=theano.config.floatX)
                z_mu = z.sum(axis=[0,1]) / z_true
                z_sigma = T.sqrt(((z - z_mu[None, None, :])**2).sum(axis=[0,1]) / z_true)
                if running_average:
                    running_mu = theano.clone(self.mu, share_inputs=False)
                    running_mu.default_update = (self.rho * self.mu + (1 - self.rho) * z_mu)
                    running_sigma = theano.clone(self.sigma, share_inputs=False)
                    running_sigma.default_update = (self.rho * self.sigma + (1 - self.rho) * z_sigma)
                    z_mu += 0 * running_mu
                    z_sigma += 0 * running_sigma
                else:
                    mu = theano.clone(self.mu, share_inputs=False)
                    mu.default_update = z_mu
                    sigma = theano.clone(self.sigma, share_inputs=False)
                    sigma.default_update = z_sigma
                    z_mu += 0 * mu
                    z_sigma += 0 * sigma
            else:
                z_mu = self.mu
                z_sigma = self.sigma
            z_sigma += self.eps
            z = (z - z_mu[None, None, :]) * (tparams['gamma_'+self.name] / z_sigma)[None, None, :] + tparams['beta_'+self.name][None, None, :]

        z = self.nonlin(z) + self.cons
        z.name = self.name

        return z

    def initialize(self):

        params = OrderedDict()

        for parname, parout in self.parent.items():
            W_shape = (parout, self.nout)
            W_name = 'W_' + parname + '__' + self.name
            params[W_name] = self.init_W.get(W_shape)

        if self.use_bias:
            params['b_'+self.name] = self.init_b.get(self.nout)

        params['beta_'+self.name] = InitCell('zeros').get(self.nout)
        params['gamma_'+self.name] = InitCell('rand', low=0.95, high=1.05).get(self.nout)

        return params


class BatchNormLSTM(RecurrentLayer):
    """
    Batch normalization long short-term memory

    Parameters
    ----------
    .. todo::
    """
    def get_init_state(self, batch_size):

        state = T.zeros((batch_size, 2*self.nout), dtype=theano.config.floatX)
        state = T.unbroadcast(state, *range(state.ndim))

        return state

    def fprop(self, XH, tparams, time_step=1, mask=None, z_mu=None, z_var=None, test=0):

        # XH is a list of inputs: [state_belows, state_befores]
        # each state vector is: [state_before; cell_before]
        # Hence, you use h[:, :self.nout] to compute recurrent term
        X, H = XH

        if len(X) != len(self.parent):
            raise AttributeError("The number of inputs doesn't match "
                                 "with the number of parents.")

        if len(H) != len(self.recurrent):
            raise AttributeError("The number of inputs doesn't match "
                                 "with the number of recurrents.")

        # The index of self recurrence is 0
        z_t = H[0]
        z = T.zeros((X[0].shape[0], 4*self.nout), dtype=theano.config.floatX)

        for x, (parname, parout) in izip(X, self.parent.items()):
            W = tparams['W_'+parname+'__'+self.name]

            if x.ndim == 1:
                if 'int' not in x.dtype:
                    x = T.cast(x, 'int64')
                z += W[x]
            else:
                z += T.dot(x[:, :parout], W)

        for h, (recname, recout) in izip(H, self.recurrent.items()):
            U = tparams['U_'+recname+'__'+self.name]
            z += T.dot(h[:, :recout], U)

        z += tparams['b_'+self.name]

        if test:
            z_mu_t = z_mu
            z_var_t = z_var
        else:
            #z_mu_t = z_mu + ((z - z_mu[None, :]) * mask[:, None]).sum(axis=0) / mask.sum() / T.cast(time_step, dtype=theano.config.floatX)
            #z_var_t = z_var + (z - z_mu[None, :]) * (z - z_mu_t[None, :]) * mask[:, None]).sum(axis=0) / mask.sum() + 1e-15
            #z_mu_t = z_mu + (((z - z_mu[None, :]) * mask[:, None]).sum(axis=0) + 1e-15) / (mask.sum() + 1e-15) / T.cast(time_step, dtype=theano.config.floatX)
            #z_var_t = z_var + (((z - z_mu[None, :]) * (z - z_mu_t[None, :]) * mask[:, None]).sum(axis=0) + 1e-15) / (mask.sum() + 1e-15)


            #z_mu_t = z_mu + (((z - z_mu[None, :]) / T.cast(time_step, dtype=theano.config.floatX)) * mask[:, None]).sum(axis=0) / mask.sum()
            #z_var_t = z_var + ((z - z_mu[None, :]) * (z - z_mu_t[None, :]) * mask[:, None]).sum(axis=0) / mask.sum()
            z_mu_t = T.switch(T.cast(T.eq(time_step, 1), 'int32'),
                                (z * mask[:, None]).sum(axis=0) / mask.sum(),
                                z_mu + (((z - z_mu[None, :]) / T.cast(time_step, dtype=theano.config.floatX)) * mask[:, None]).sum(axis=0) / mask.sum())
            z_var_t = T.switch(T.cast(T.eq(time_step, 1), 'int32'),
                               z_var,
                               z_var + ((z - z_mu[None, :]) * (z - z_mu_t[None, :]) * mask[:, None]).sum(axis=0) / mask.sum())

        z = T.switch(T.cast(T.eq(time_step, 1), 'int32'),
                     z - z_mu_t[None, :],
                     (z - z_mu_t[None, :]) / (T.sqrt(z_var_t)[None, :] + 1e-6))
        z = tparams['gamma_'+self.name][None, :] * z + tparams['beta_'+self.name][None, :]

        # Compute activations of gating units
        i_on = T.nnet.sigmoid(z[:, self.nout:2*self.nout])
        f_on = T.nnet.sigmoid(z[:, 2*self.nout:3*self.nout])
        o_on = T.nnet.sigmoid(z[:, 3*self.nout:])

        # Update hidden & cell states
        z_t = T.set_subtensor(
            z_t[:, self.nout:],
            f_on * z_t[:, self.nout:] +
            i_on * self.nonlin(z[:, :self.nout])
        )

        z_t = T.set_subtensor(
            z_t[:, :self.nout],
            o_on * self.nonlin(z_t[:, self.nout:])
        )

        z_t.name = self.name

        return z_t, z_mu_t, z_var_t

    def initialize(self):

        params = OrderedDict()
        N = self.nout

        for parname, parout in self.parent.items():
            W_shape = (parout, 4*N)
            W_name = 'W_' + parname + '__' + self.name
            params[W_name] = self.init_W.get(W_shape)

        for recname, recout in self.recurrent.items():
            M = recout
            U = self.init_U.ortho((M, N))

            for j in xrange(3):
                U = np.concatenate([U, self.init_U.ortho((M, N))], axis=-1)
            U_name = 'U_'+recname+'__'+self.name
            params[U_name] = U

        params['b_'+self.name] = self.init_b.get(4*N)
        params['beta_'+self.name] = InitCell('zeros').get(4*N)
        params['gamma_'+self.name] = InitCell('ones').get(4*N)

        return params
