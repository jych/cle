import ipdb
import numpy as np
import theano
import theano.tensor as T

from cle.cle.layers import StemCell, InitCell
from cle.cle.utils import tolist
from cle.cle.utils.op import add_noise

from itertools import izip

from theano.compat.python2x import OrderedDict


class RecurrentLayer(StemCell):
    """
    Abstract class for recurrent layers

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 recurrent=[],
                 recurrent_dim=[],
                 self_recurrent=1,
                 init_U=InitCell('ortho'),
                 **kwargs):
        super(RecurrentLayer, self).__init__(**kwargs)
        self.recurrent = OrderedDict()

        if self_recurrent:
            self.recurrent[self.name] = self.nout

        recurrent_dim = tolist(recurrent_dim)

        for i, rec in enumerate(tolist(recurrent)):
            if len(recurrent_dim) != 0:
                self.recurrent[rec] = recurrent_dim[i]
            else:
                self.recurrent[rec] = None

        self.init_U = init_U

    def get_init_state(self, batch_size):

        state = T.zeros((batch_size, self.nout), dtype=theano.config.floatX)
        state = T.unbroadcast(state, *range(state.ndim))

        return state

    def initialize(self):

        params = super(RecurrentLayer, self).initialize()

        for recname, recout in self.recurrent.items():
            U_shape = (recout, self.nout)
            U_name = 'U_'+recname+'__'+self.name
            params[U_name] = self.init_U.get(U_shape)

        return params


class SimpleRecurrent(RecurrentLayer):
    """
    Vanilla recurrent layer

    Parameters
    ----------
    .. todo::
    """
    def fprop(self, XH, tparams):

        # XH is a list of inputs: [state_belows, state_befores]
        X, H = XH

        if len(X) != len(self.parent):
            raise AttributeError("The number of inputs doesn't match "
                                 "with the number of parents.")

        if len(H) != len(self.recurrent):
            raise AttributeError("The number of inputs doesn't match "
                                 "with the number of recurrents.")

        z = T.zeros((X[0].shape[0], self.nout), dtype=theano.config.floatX)

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
        z = self.nonlin(z)
        z.name = self.name

        return z


class LSTM(RecurrentLayer):
    """
    Long short-term memory

    Parameters
    ----------
    .. todo::
    """
    def get_init_state(self, batch_size):

        state = T.zeros((batch_size, 2*self.nout), dtype=theano.config.floatX)
        state = T.unbroadcast(state, *range(state.ndim))

        return state

    def fprop(self, XH, tparams):

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

        # Compute activations of gating units
        i_t = T.nnet.sigmoid(z[:, self.nout:2*self.nout])
        f_t = T.nnet.sigmoid(z[:, 2*self.nout:3*self.nout])
        o_t = T.nnet.sigmoid(z[:, 3*self.nout:])

        # Update hidden & cell states
        z_t = T.set_subtensor(
            z_t[:, self.nout:],
            f_t * z_t[:, self.nout:] +
            i_t * self.nonlin(z[:, :self.nout])
        )

        z_t = T.set_subtensor(
            z_t[:, :self.nout],
            o_t * self.nonlin(z_t[:, self.nout:])
        )

        z_t.name = self.name

        return z_t

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

        return params


class GFLSTM(LSTM):
    """
    Gated feedback long short-term memory

    Parameters
    ----------
    .. todo::
    """
    def fprop(self, XH, tparams):

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
        Nm = len(self.recurrent)
        z = T.zeros((X[0].shape[0], 4*self.nout+Nm), dtype=theano.config.floatX)

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
            z = T.inc_subtensor(
                z[:, self.nout:],
                T.dot(h[:, :recout], U[:, self.nout:])
            )

        z += tparams['b_'+self.name]

        # Compute activations of gating units
        i_t = T.nnet.sigmoid(z[:, self.nout:2*self.nout])
        f_t = T.nnet.sigmoid(z[:, 2*self.nout:3*self.nout])
        o_t = T.nnet.sigmoid(z[:, 3*self.nout:4*self.nout])
        gron = T.nnet.sigmoid(z[:, 4*self.nout:])
        c_t = z[:, :self.nout]

        for i, (h, (recname, recout)) in\
            enumerate(izip(H, self.recurrent.items())):
            gated_h = h[:, :recout] * gron[:, i].dimshuffle(0, 'x')
            U = tparams['U_'+recname+'__'+self.name]
            c_t += T.dot(gated_h, U[:, :self.nout])

        # Update hidden & cell states
        z_t = T.set_subtensor(
            z_t[:, self.nout:],
            f_t * z_t[:, self.nout:] +
            i_t * self.nonlin(c_t)
        )

        z_t = T.set_subtensor(
            z_t[:, :self.nout],
            o_t * self.nonlin(z_t[:, self.nout:])
        )

        z_t.name = self.name

        return z_t

    def initialize(self):

        params = OrderedDict()
        N = self.nout
        Nm = len(self.recurrent)

        for parname, parout in self.parent.items():
            W_shape = (parout, 4*N+Nm)
            W_name = 'W_' + parname + '__' + self.name
            params[W_name] = self.init_W.get(W_shape)

        for recname, recout in self.recurrent.items():
            M = recout
            U = self.init_U.ortho((M, N))

            for j in xrange(3):
                U = np.concatenate([U, self.init_U.ortho((M, N))], axis=-1)

            U = np.concatenate([U, self.init_U.rand((M, Nm))], axis=-1)
            U_name = 'U_'+recname+'__'+self.name
            params[U_name] = U

        params['b_'+self.name] = self.init_b.get(4*N+Nm)

        return params


class GRU(RecurrentLayer):
    """
    Gated recurrent unit

    Parameters
    ----------
    .. todo::
    """
    def fprop(self, XH, tparams):

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
        z_tm1 = H[0]
        z = T.zeros((X[0].shape[0], 3*self.nout), dtype=theano.config.floatX)

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
            z = T.inc_subtensor(
                z[:, self.nout:],
                T.dot(h[:, :recout], U[:, self.nout:])
            )

        z += tparams['b_'+self.name]

        # Compute activations of gating units
        r_on = T.nnet.sigmoid(z[:, self.nout:2*self.nout])
        u_on = T.nnet.sigmoid(z[:, 2*self.nout:])

        # Update hidden & cell states
        c_t = T.zeros_like(z_tm1)

        for h, (recname, recout) in izip(H, self.recurrent.items()):
            U = tparams['U_'+recname+'__'+self.name]
            c_t += T.dot(h[:, :recout], U[:, :self.nout])

        z_t = T.tanh(z[:, :self.nout] + r_on * c_t)
        z_t = u_on * z_tm1 + (1. - u_on) * z_t
        z_t.name = self.name

        return z_t

    def initialize(self):

        params = OrderedDict()
        N = self.nout

        for parname, parout in self.parent.items():
            W_shape = (parout, 3*N)
            W_name = 'W_' + parname + '__' + self.name
            params[W_name] = self.init_W.get(W_shape)

        for recname, recout in self.recurrent.items():
            M = recout
            U = self.init_U.ortho((M, N))

            for j in xrange(2):
                U = np.concatenate([U, self.init_U.ortho((M, N))], axis=-1)

            U_name = 'U_'+recname+'__'+self.name
            params[U_name] = U

        params['b_'+self.name] = self.init_b.get(3*N)

        return params


class GRU2(GRU):
    """
    Gated recurrent unit with different implementation
    \tilde{h}_t = Wx + U(r \odot h_{t-1})

    Parameters
    ----------
    .. todo::
    """
    def fprop(self, XH, tparams):

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
        z_tm1 = H[0]
        z = T.zeros((X[0].shape[0], 3*self.nout), dtype=theano.config.floatX)

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
            z = T.inc_subtensor(
                z[:, self.nout:],
                T.dot(h[:, :recout], U[:, self.nout:])
            )

        z += tparams['b_'+self.name]

        # Compute activations of gating units
        r_on = T.nnet.sigmoid(z[:, self.nout:2*self.nout])
        u_on = T.nnet.sigmoid(z[:, 2*self.nout:])

        # Update hidden & cell states
        c_t = T.zeros_like(z_tm1)

        for h, (recname, recout) in izip(H, self.recurrent.items()):
            U = tparams['U_'+recname+'__'+self.name]
            c_t += T.dot(r_on * h[:, :recout], U[:, :self.nout])

        z_t = T.tanh(z[:, :self.nout] + c_t)
        z_t = u_on * z_tm1 + (1. - u_on) * z_t
        z_t.name = self.name

        return z_t


class GFGRU(GRU):
    """
    Long short-term memory

    Parameters
    ----------
    .. todo::
    """
    def fprop(self, XH, tparams):

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
        z_tm1 = H[0]
        Nm = len(self.recurrent)
        z = T.zeros((X[0].shape[0], 3*self.nout+Nm), dtype=theano.config.floatX)

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
            z = T.inc_subtensor(
                z[:, self.nout:],
                T.dot(h[:, :recout], U[:, self.nout:])
            )

        z += tparams['b_'+self.name]

        # Compute activations of gating units
        r_on = T.nnet.sigmoid(z[:, self.nout:2*self.nout])
        u_on = T.nnet.sigmoid(z[:, 2*self.nout:3*self.nout])
        gron = T.nnet.sigmoid(z[:, 3*self.nout:])

        # Update hidden & cell states
        c_t = T.zeros_like(z_tm1)

        for i, (h, (recname, recout)) in\
            enumerate(izip(H, self.recurrent.items())):
            gated_h = h[:, :recout] * gron[:, i].dimshuffle(0, 'x')
            U = tparams['U_'+recname+'__'+self.name]
            c_t += T.dot(gated_h, U[:, :self.nout])

        z_t = T.tanh(z[:, :self.nout] + r_on * c_t)
        z_t = u_on * z_tm1 + (1. - u_on) * z_t
        z_t.name = self.name

        return z_t

    def initialize(self):

        params = OrderedDict()
        N = self.nout
        Nm = len(self.recurrent)

        for parname, parout in self.parent.items():
            W_shape = (parout, 3*N+Nm)
            W_name = 'W_' + parname + '__' + self.name
            params[W_name] = self.init_W.get(W_shape)

        for recname, recout in self.recurrent.items():
            M = recout
            U = self.init_U.ortho((M, N))

            for j in xrange(2):
                U = np.concatenate([U, self.init_U.ortho((M, N))], axis=-1)
            U = np.concatenate([U, self.init_U.rand((M, Nm))], axis=-1)
            U_name = 'U_'+recname+'__'+self.name
            params[U_name] = U

        params['b_'+self.name] = self.init_b.get(3*N+Nm)

        return params
