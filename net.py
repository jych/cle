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

    def get_params(self):
        return flatten([node.get_params() for node in self.nodes])

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
