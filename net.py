import cPickle, gzip, os, sys, time, re, ipdb
import numpy as np

from theano.compat.python2x import OrderedDict
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from util import *
from layer import *


class Net(Layer):
    """
    Abstract class for networks

    Parameters
    ----------
    todo..
    """
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

<<<<<<< HEAD
class FeedForwardNet(Net):
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


class FullyConnectedLayer(Layer):
    # Implementations of Layer
    def __init__(self,
                 name,
                 n_in,
                 n_out,
                 unit='relu',
                 init_W=ParamInit('randn'),
                 init_b=ParamInit('zeros')):
        self.__dict__.update(locals())
        del self.self
        self.W = init_W.get(n_in, n_out)
        self.b = init_b.get(n_out)

    @property
    def params(self):
        return [self.W, self.b]
=======
    def get_params(self):
        return flatten([node.get_params() for node in self.nodes])
>>>>>>> 6ca993fc6736323ec3e8bbd6abdbd96dc4d03a8a

    def fprop(self, x):
        ipdb.set_trace()
        evals = []
        z = self.nodes['cost'].fprop(self.nodes[self.edges['cost']])

        return x

    def compute_cost(self, x, y):
        y_hat = self.fprop(x)
        return self.cost.fprop(y, y_hat)

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)
