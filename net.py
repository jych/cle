import cPickle, gzip, os, sys, time, re
import numpy as np
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.compat.python2x import OrderedDict
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from util import *


class ParamInit(object):
    def __init__(self,
                 init_type='randn',
                 mean=0.,
                 stddev=0.01,
                 low=-0.1,
                 high=0.1):
        self.initializer = {
            'rand': lambda x: np.random.uniform(low=low,
                                                high=high,
                                                size=x.shape),
            'randn': lambda x: np.random.normal(loc=mean,
                                                scale=stddev,
                                                size=x.shape),
            'zeros': lambda x: np.zeros(x.shape),
            'const': lambda x: np.zeros(x.shape) + mean,
            'ortho': lambda x: scipy.linalg.orth(
                np.random.normal(loc=mean, scale=stddev, size=x.shape))
        }[init_type]

    def set(self, sharedX_var):
        numpy_var = sharedX_var.get_value()
        numpy_var = castX(self.initializer(numpy_var))
        sharedX_var.set_value(numpy_var)

    def get(self, *shape):
        return sharedX(self.initializer(np.zeros(shape)))


class Nonlin(object):

    def which_nonlin(self, nonlin):

        return getattr(self, nonlin)

    def linear(self, z):
        return z

    def relu (self, z):
        return z * (z > 0.)

    def sigmoid(self, z):
        return T.nnet.sigmoid(z)

    def softmax(self, z):
        return T.nnet.softmax(z)

    def tanh(self, z):
        return T.nnet.tanh(z)


class Layer(object, Nonlin):
    # Abstract class for layers
    def __init__(self):
        pass

    @property
    def params(self):
        return []

    def fprop(self, x=None):
        return x


class FullyConnectedLayer(Layer):
    # Implementations of Layer
    def __init__(self,
                 name,
                 n_in,
                 n_out,
                 unit='relu',
                 init_W=ParamInit('randn'),
                 init_b=ParamInit('zeros')):

        self.nonlin = self.which_nonlin(unit)
        self.W = init_W.get(n_in, n_out)
        self.b = init_b.get(n_out)

    @property
    def params(self):
        return [self.W, self.b]

    def fprop(self, x):
        z = T.dot(x, self.W) + self.b
        z = self.nonlin(z)
        return z


#class DataLayer(Layer):

#    def __init__(self):
#        self.sym = T.fmatrix()

#    def fprop(self):
#        return self.sym


#class DesignMatrixDataLayer(DataLayer):

#    def __init__(self, name, np_data, batch_size=None):
#        self.name = name
#        self.n_data = np_data.shape[0]
#        self.batch_size = batch_size if batch_size is not None else self.n_data
#        self.sharedX_data = sharedX(np_data)
#        self.sym_data = T.fmatrix()

#    def fprop(self, x=None):
#        return self.sym_data

#    def get_batch(self, i):
#        i = i % (self.n_data / self.batch_size + 1)
#        return self.sharedX_data[i*self.batch_size: (i+1)*self.batch_size]


class CostLayer(Layer):

    def __init__(self, name):
        pass

    def fprop(self, x, y):
        pass


class MulticlassCostLayer(CostLayer):

    def __init__(self, name, target):
        self.name = name
        self.target = target

    def fprop(self, p, y=None):
        y = self.target if y is None else y
        return - T.sum(y * T.log(p)) / p.shape[0]


class Net(Layer):
    # Abstract class for networks
    def __init__(self, layers, edges):
        pass

    @property
    def params(self):
        pass

    def fprop(self, x=None):
        pass


class SeqNet(Net):
    # Temporary class
    def __init__(self, name, *layers):
        self.name = name
        self.layers = layers

    @property
    def params(self):
        return flatten([layer.params for layer in self.layers])

    def fprop(self, x=None):
        for layer in self.layers:
            x = layer.fprop(x)
        return x


