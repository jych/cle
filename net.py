import cPickle, gzip, os, sys, time, re
import numpy as np

from theano.compat.python2x import OrderedDict
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from util import *
from layer import *


#class Net(Layer):
#    """
#    Abstract class for networks

#    Parameters
#    ----------
#    todo..
#    """
#    def __init__(self, layers, edges):
        pass

#    @property
#    def params(self):
#        pass

#    def fprop(self, x=None):
#        pass


#class SeqNet(Net):

#    def __init__(self, name, *layers):
#        self.name = name
#        self.layers = layers

#    @property
#    def params(self):
#        return flatten([layer.params for layer in self.layers])

#    def fprop(self, x=None):
#        for layer in self.layers:
#            x = layer.fprop(x)
#        return x


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

    @property
    def params(self):
        return []

    def fprop(self, x):

        

        return x

    @classmethod
    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)

