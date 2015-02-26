import ipdb
import numpy as np

from cle.cle.util import flatten, tolist, topological_sort
from theano.compat.python2x import OrderedDict


class Net(object):
    """
    Abstract class for networks

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, nodes, inputs=None):
        if inputs is None:
            inputs = self.set_inputs(nodes)
        self.inputs = inputs
        self.set_graph(nodes)
        self.set_nodes(nodes)
        self.initialize()
        self.params = self.get_params()

    def set_inputs(self, nodes):
        inputs = []
        for node in nodes:
            if node.isroot:
                inputs.append(node.out)
        return inputs

    def set_nodes(self, nodes):
        self.nodes = {}
        for node in nodes:
            self.nodes[node.name] = node

    def initialize(self):
        for node in self.nodes.values():
            node.initialize()

    def set_graph(self, nodes):
        self.graph = {}
        for node in nodes:
            if not node.isroot:
                parent = node.parent
                for par_node in tolist(parent):
                    self.graph[par_node.name] = node.name

    def build_graph(self):
        sorted_nodes = topological_sort(self.graph)
        while sorted_nodes:
            node = sorted_nodes.popleft()
            if self.nodes[node].isroot:
                continue
            parent = self.nodes[node].parent
            inp = []
            for par in parent:
                inp.append(par.out)
            self.nodes[node].out = self.nodes[node].fprop(inp)

    def build_recurrent_graph(self):
        sorted_nodes = topological_sort(self.graph)
        # call get_init_state() here, which is basically Theano variables
        # we do not explicitly set these things outside which will just
        # increase confusion
        # A recurrent layer should have a list of T.zeros for each
        # recurrent object fed-in
        #...
        #>>> pseudo code
        #init_node = node_which_is_rec.get_init_state()



        while sorted_nodes:
            node = sorted_nodes.popleft()
            if self.nodes[node].isroot:
                continue
            parent = self.nodes[node].parent
            inp = []
            for par in parent:
                inp.append(par.out)
            self.nodes[node].out = self.nodes[node].fprop(inp)

    def build_scan_graph(self, fn, seq_args, output_info, nonseq_args):
        # fn should be the model_last_layer.out
        return theano.scan()

    def get_params(self):
        #return flatten([node.get_params() for node in self.nodes.values()])
        #params = OrderedDict()
        #for node in self.nodes.values():
        #    for key, value in node.get_params().items():
        #        params[key] = value
        #return params
        return flatten([node.get_params().values()
                        for node in self.nodes.values()])

    def get_inputs(self):
        return self.inputs
