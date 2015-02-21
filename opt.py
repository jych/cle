import ipdb
import numpy as np
import theano.tensor as T

from itertools import izip
from theano.compat.python2x import OrderedDict
from util import *


# Things to add
# 1. Simple Momentum
class Optimizer(object):
    def __init__(self):
        pass


    def clip_gradients(self, grads):
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


class RMSProp(Optimizer):
    """
    .. todo::

        WRITEME
    """
    def __init__(self, 
                 learning_rate, 
                 init_momentum=0.9, 
                 averaging_coeff=0.95, 
                 stabilizer=0.0001,
                 gradient_clipping=False):
        self.__dict__.update(locals())
        del self.self
        self.momentum = init_momentum

    def get_updates(self, cost, params):
        """
        .. todo::

            WRITEME
        """
        grads = OrderedDict(izip(params, T.grad(cost, params)))
        updates = OrderedDict()

        if self.gradient_clipping:
            grads = self.clip_gradients(grads)
 
        for param in grads.keys():
            inc = sharedX(param.get_value() * 0.)
            avg_grad = sharedX(np.zeros_like(param.get_value()))
            avg_grad_sqr = sharedX(np.zeros_like(param.get_value()))

            new_avg_grad = self.averaging_coeff * avg_grad\
                + (1 - self.averaging_coeff) * grads[param]
            new_avg_grad_sqr = self.averaging_coeff * avg_grad_sqr\
                + (1 - self.averaging_coeff) * grads[param]**2

            normalized_grad = grads[param] /\
                T.sqrt(new_avg_grad_sqr - new_avg_grad**2 + self.stabilizer)
            updated_inc = self.momentum * inc - self.learning_rate * normalized_grad

            updates[avg_grad] = new_avg_grad
            updates[avg_grad_sqr] = new_avg_grad_sqr
            updates[inc] = updated_inc
            updates[param] = param + updated_inc

        return updates


class Adam(Optimizer):
    """
    The MIT License (MIT)

    Copyright (c) 2015 Alec Radford

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    def __init__(self,
                 learning_rate,
                 init_momentum=0.9,
                 averaging_coeff=0.99,
                 stabilizer=1e-4,
                 gradient_clipping=False):
        self.__dict__.update(locals())
        del self.self
        self.momentum = init_momentum

    def get_updates(self, cost, params):
        """
        .. todo::

            WRITEME
        """
        grads = OrderedDict(izip(params, T.grad(cost, params)))
        updates = OrderedDict()
        velocity = OrderedDict()
        counter = sharedX(0, 'counter')

        if self.gradient_clipping:
            grads = self.clip_gradients(grads)

        for param in grads.keys():
            avg_grad_sqr = sharedX(np.zeros_like(param.get_value()))
            velocity[param] = sharedX(np.zeros_like(param.get_value()))

            next_counter = counter + 1.

            fix_first_moment = 1. - self.momentum**next_counter
            fix_second_moment = 1. - self.averaging_coeff**next_counter

            if param.name is not None:
                avg_grad_sqr.name = 'avg_grad_sqr_' + param.name

            new_avg_grad_sqr = self.averaging_coeff*avg_grad_sqr \
                + (1 - self.averaging_coeff)*T.sqr(grads[param])

            rms_grad_t = T.sqrt(new_avg_grad_sqr)
            rms_grad_t = T.maximum(rms_grad_t, self.stabilizer)
            new_velocity = self.momentum * velocity[param] \
                - (1 - self.momentum) * grads[param]

            normalized_velocity = (new_velocity * T.sqrt(fix_second_moment)) \
                / (rms_grad_t * fix_first_moment)

            updates[avg_grad_sqr] = new_avg_grad_sqr
            updates[velocity[param]] = new_velocity
            updates[param] = param + learning_rate * normalized_velocity

        updates[counter] = counter + 1

        return updates
