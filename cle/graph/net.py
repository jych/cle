import ipdb
import numpy as np
import theano
import theano.tensor as T

from collections import OrderedDict
from cle.cle.util import flatten, tolist, topological_sort


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
            if not node.isroot and not node.istarget:
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

    def build_recurrent_graph(self, **kwargs):
        sorted_nodes = topological_sort(self.graph)
        # call get_init_state() here, which is basically Theano variables
        # we do not explicitly set these things outside which will just
        # increase confusion
        # A recurrent layer should have a list of T.zeros for each
        # recurrent object fed-in
        #...
        #>>> pseudo code
        #init_node = node_which_is_rec.get_init_state()

        # Pre scan, set sequences, output infos, non-sequences
        #output_info = OrderedDict()
        output_info = []
        seq_args = OrderedDict()
        nonseq_args = OrderedDict()
        targets = OrderedDict()

        # pop cost function
        cost_obj = sorted_nodes.pop('cost')

        for node in self.nodes:
            if hasattr(node, 'isroot'):
                if node.isroot:
                    seq_args.update(node)
            if hasattr(node, 'istarget'):
                if node.istarget:
                    targets.update(node)
            if hasattr(node, 'get_init_state'):
                state = node.get_init_state()
                output_info.append(state)
        nstate = len(output_info)

        n_steps = kwargs.pop('n_steps', None)
        reverse = kwargs.pop('reverse', False)
        for key, value in kwargs.items():
            output_info[key] = value        
        nNone = len(output_info) - nstate
        output_info = flatten(output_info + [None] * nNone)

        def scan_fn(*args):
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





        result, updates = theano.scan(
            fn=scan_fn,
            sequences=list(seq_args.values()),
            outputs_info=outputs_info,
            non_sequences=list(nonseq_args.values()),
            n_steps=n_steps,
            go_backwards=reverse)
        result = tolist(result)

        # Post scan, add cost
        inp = flatten(list([result, targets))
        cost = cost_obj(inp)
        ipdb.set_trace()
        return cost

    def get_params(self):
        #return flatten([node.get_params() for node in self.nodes.values()])
        return flatten([node.get_params().values()
                        for node in self.nodes.values()])

    def get_inputs(self):
        return self.inputs
