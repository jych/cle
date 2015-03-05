import theano.tensor as T

from theano.compat.python2x import OrderedDict
from cle.cle.utils import sharedX


class Optimizer(object):
    def __init__(self, lr):
        """
        .. todo::

            WRITEME
        """
        self.lr = sharedX(lr)
        self.updates = []

    def get_updates(self):
        """
        .. todo::

            WRITEME
        """
        pass


class Momentum(Optimizer):
    """
    .. todo::

        WRITEME
    """
    def __init__(self,
                 mom=0.9,
                 nesterov=False,
                 **kwargs):
        self.__dict__.update(locals())
        del self.self
        super(Momentum, self).__init__(**kwargs)

    def get_updates(self, grads):
        """
        .. todo::

            WRITEME
        """
        updates = OrderedDict()
        for p, g in grads.items():
            u = sharedX(param.get_value() * 0.)
            u_t = self.mom * u - self.lr * g
            if self.nesterov:
                u_t = self.mom * u_t - self.lr * g
            p_t = p + u_t
            updates[u] = u_t
            updates[p] = p_t
        return updates


class RMSProp(Optimizer):
    """
    .. todo::

        WRITEME
    """
    def __init__(self, mom=0.9, coeff=0.95, e=1e-4, **kwargs):
        self.__dict__.update(locals())
        del self.self
        super(RMSProp, self).__init__(**kwargs)

    def get_updates(self, grads):
        """
        .. todo::

            WRITEME
        """
        updates = OrderedDict()
        for p, g in grads.items():
            u = sharedX(p.get_value() * 0.)
            avg_grad = sharedX(p.get_value() * 0.)
            sqr_grad = sharedX(p.get_value() * 0.)
            avg_grad_t = self.coeff * avg_grad + (1 - self.coeff) * g
            sqr_grad_t = self.coeff * sqr_grad + (1 - self.coeff) * g**2
            g_t = g / T.sqrt(sqr_grad_t - avg_grad_t**2 + self.e)
            u_t = self.mom * u - self.lr * g_t
            p_t = p + u_t
            updates[avg_grad] = avg_grad_t
            updates[sqr_grad] = sqr_grad_t
            updates[u] = u_t
            updates[p] = p_t
        return updates


class Adam(Optimizer):
    """
    .. todo::

        WRITEME
    """
    def __init__(self, b1=0.1, b2=0.001, e=1e-8, **kwargs):
        self.__dict__.update(locals())
        del self.self
        super(Adam, self).__init__(**kwargs)

    def get_updates(self, grads):
        """
        .. todo::

            WRITEME
        """
        updates = OrderedDict()
        m = OrderedDict()
        v = OrderedDict()
        cnt = sharedX(0, 'counter')
        for p, g in grads.items():
            m = sharedX(p.get_value() * 0.)
            v = sharedX(p.get_value() * 0.)
            m_t = (1. - self.b1) * m + self.b1 * g
            v_t = (1. - self.b2) * v + self.b2 * T.sqr(g)
            m_t_hat = m_t / (1. - (1. - self.b1)**(cnt + 1))
            v_t_hat = v_t / (1. - (1. - self.b2)**(cnt + 1))
            g_t = m_t_hat / (T.sqrt(v_t_hat) + self.e)
            p_t = p - self.lr * g_t
            updates[m] = m_t
            updates[v] = v_t
            updates[p] = p_t
        updates[cnt] = cnt + 1
        return updates
