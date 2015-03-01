import ipdb
import numpy as np
import theano.tensor as T

from itertools import izip
from cle.cle.layers import StemCell, RandomCell, InitCell
from cle.cle.util import tolist
from theano.compat.python2x import OrderedDict


class RecurrentLayer(StemCell):
    """
    Abstract class for recurrent layers

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 batch_size,
                 recurrent=[],
                 init_U=InitCell('ortho'),
                 **kwargs):
        super(RecurrentLayer, self).__init__(**kwargs)
        self.recurrent = tolist(recurrent)
        self.recurrent.append(self)
        self.batch_size = batch_size
        self.init_U = init_U
        self.init_states = OrderedDict()

    def get_init_state(self):
        state = T.zeros((self.batch_size, self.nout))
        state = T.unbroadcast(state, *range(state.ndim))
        return state

    def initialize(self):
        super(RecurrentLayer, self).initialize()
        for i, recurrent in enumerate(self.recurrent):
            self.alloc(self.init_U.get((recurrent.nout, self.nout),
                                      'U_'+recurrent.name+self.name))


class SimpleRecurrent(RecurrentLayer):
    """
    Vanilla recurrent layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 unit='tanh',
                 **kwargs):
        super(SimpleRecurrent, self).__init__(**kwargs)
        self.nonlin = self.which_nonlin(unit)

    def fprop(self, xh):
        # xh is a list of inputs: [state_belows, state_befores]
        xs, hs = xh
        z = T.zeros(self.nout)
        for x, parent in izip(xs, self.parent):
            z += T.dot(x, self.params['W_'+parent.name+self.name])
        for h, recurrent in izip(hs, self.recurrent):
            z += T.dot(h, self.params['U_'+recurrent.name+self.name])
        z += self.params['b_'+self.name]
        z = self.nonlin(z)
        z.name = self.name
        return z


class LSTM(SimpleRecurrent):
    """
    Long short-term memory

    Parameters
    ----------
    .. todo::
    """
    def get_init_state(self):
        state = T.zeros((self.batch_size, 2*self.nout))
        state = T.unbroadcast(state, *range(state.ndim))
        return state
  
    def fprop(self, xh):
        # xh is a list of inputs: [state_belows, state_befores]
        # each state vector is: [state_before; cell_before]
        # Hence, you use h[:, :self.nout] to compute recurrent term
        xs, hs = xh
        # The index of self recurrence is 0
        z_t = hs[0]
        z = T.zeros((self.batch_size, 4*self.nout))
        for x, parent in izip(xs, self.parent):
            W = self.params['W_'+parent.name+self.name]
            z += T.dot(x[:, :parent.nout], W)
        for h, recurrent in izip(hs, self.recurrent):
            U = self.params['U_'+recurrent.name+self.name]
            z += T.dot(h[:, :recurrent.nout], U)
        z += self.params['b_'+self.name]
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
        return z_t

    def initialize(self):
        N = self.nout
        for i, parent in enumerate(self.parent):
            self.alloc(self.init_W.get((parent.nout, 4*N),
                                       'W_'+parent.name+self.name))
        for i, recurrent in enumerate(self.recurrent):
            M = recurrent.nout
            U = self.init_U.ortho((M, N))
            for j in xrange(3):
                U = np.concatenate([U, self.init_U.ortho((M, N))], axis=-1)
            U = self.init_U.setX(U, 'U_'+recurrent.name+self.name)
            self.alloc(U)
        self.alloc(self.init_b.get(4*N, 'b_'+self.name))


class GFLSTM(LSTM):
    """
    Gated Feedback Long short-term memory

    Parameters
    ----------
    .. todo::
    """
    def fprop(self, xh):
        # xh is a list of inputs: [state_belows, state_befores]
        # each state vector is: [state_before; cell_before]
        # Hence, you use h[:, :self.nout] to compute recurrent term
        xs, hs = xh
        # The index of self recurrence is 0
        z_t = hs[0]
        Nm = len(self.recurrent)
        z = T.zeros((self.batch_size, 4*self.nout+Nm))
        for x, parent in izip(xs, self.parent):
            W = self.params['W_'+parent.name+self.name]
            z += T.dot(x[:, :parent.nout], W)
        for h, recurrent in izip(hs, self.recurrent):
            U = self.params['U_'+recurrent.name+self.name]
            z = T.inc_subtensor(
                z[:, self.nout:],
                T.dot(h[:, :recurrent.nout], U[:, self.nout:])
            )
        z += self.params['b_'+self.name]
        # Compute activations of gating units
        i_on = T.nnet.sigmoid(z[:, self.nout:2*self.nout])
        f_on = T.nnet.sigmoid(z[:, 2*self.nout:3*self.nout])
        o_on = T.nnet.sigmoid(z[:, 3*self.nout:4*self.nout])
        gron = T.nnet.sigmoid(z[:, 4*self.nout:])
        c_t = z[:, :self.nout]
        for i, (h, recurrent) in enumerate(izip(hs, self.recurrent)):
            gated_h = h[:, :recurrent.nout] * gron[:, i].dimshuffle(0, 'x')
            U = self.params['U_'+recurrent.name+self.name]
            c_t += T.dot(gated_h, U[:, :self.nout])
        # Update hidden & cell states
        z_t = T.set_subtensor(
            z_t[:, self.nout:],
            f_on * z_t[:, self.nout:] +
            i_on * self.nonlin(c_t)
        )
        z_t = T.set_subtensor(
            z_t[:, :self.nout],
            o_on * self.nonlin(z_t[:, self.nout:])
        )
        z_t.name = self.name
        return z_t

    def initialize(self):
        N = self.nout
        Nm = len(self.recurrent)
        for i, parent in enumerate(self.parent):
            self.alloc(self.init_W.get((parent.nout, 4*N+Nm),
                                       'W_'+parent.name+self.name))
        for i, recurrent in enumerate(self.recurrent):
            M = recurrent.nout
            U = self.init_U.ortho((M, N))
            for j in xrange(3):
                U = np.concatenate([U, self.init_U.ortho((M, N))], axis=-1)
            U = np.concatenate([U, self.init_U.rand((M, Nm))], axis=-1)
            U = self.init_U.setX(U, 'U_'+recurrent.name+self.name)
            self.alloc(U)
        self.alloc(self.init_b.get(4*N+Nm, 'b_'+self.name))
