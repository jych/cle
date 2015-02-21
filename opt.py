import numpy as np
import theano.tensor as T

from util import *


class Optimizer(object):
    def __init__(self):
        pass


class RMSProp(Optimizer):
    def __init__(self, 
                 learning_rate, 
                 momentum=0.9, 
                 averaging_coeff=0.95, 
                 stabilizer=0.0001):
        self.__dict__.update(locals())

    def updates(self, cost, params):
        updates = OrderedDict()

        for param, grad_param in zip(params, T.grad(cost, params)):

            inc = sharedX(param.get_value() * 0.)
            avg_grad = sharedX(np.zeros_like(param.get_value()))
            avg_grad_sqr = sharedX(np.zeros_like(param.get_value()))

            new_avg_grad = self.averaging_coeff * avg_grad\
                + (1 - self.averaging_coeff) * grad_param
            new_avg_grad_sqr = self.averaging_coeff * avg_grad_sqr\
                + (1 - self.averaging_coeff) * grad_param**2

            normalized_grad = grad_param /\
                T.sqrt(new_avg_grad_sqr - new_avg_grad**2 + self.stabilizer)
            updated_inc = self.momentum * inc - self.learning_rate * normalized_grad

            updates[avg_grad] = new_avg_grad
            updates[avg_grad_sqr] = new_avg_grad_sqr
            updates[inc] = updated_inc
            updates[param] = param + updated_inc

        return updates
