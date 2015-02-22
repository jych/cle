import numpy as np
import theano.tensor as T

from itertools import izip
from util import *

class GradientClipping(object):
    def __init__(self):
        """
        .. todo::

            WRITEME
        """
        self.name = 'ext_grads'

    def apply(self, grads):
        """
        .. todo::

            WRITEME
        """
        g_norm = 0.
        for grad in grads.values():
            grad /= 128
            g_norm += (grad ** 2).sum()
        not_finite = T.or_(T.isnan(g_norm), T.isinf(g_norm))
        g_norm = T.sqrt(g_norm)
        scaling_num = 5
        scaling_den = T.maximum(5, g_norm)
        for param, grad in grads.items():
            grads[param] = T.switch(not_finite,
                                    0.1 * param,
                                    grad * (scaling_num / scaling_den))

        return grads


class EpochCount(object):
    def __init__(self, num_epoch):
        """
        .. todo::

            WRITEME
        """
        self.name = 'ext_terms'
        self.num_epoch = num_epoch
        self._cnt = 0

    def validate(self):
        """
        .. todo::

            WRITEME
        """
        self._cnt += 1
        if np.mod(self._cnt, self.num_epoch)==0:
            return True
        else:
            return False
