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

        # Pre scan, set sequences, outputs info, nonsequences
        #outputs = OrderedDict()
        given_nonseq_args = kwargs.pop('nonseq_args', None)
        given_output_args = kwargs.pop('output_args', None)
        given_args = kwargs.pop('given_args', None)
        n_steps = None
        reverse = None
        seqs = []
        outputs = []
        nonseqs = []
        seq_args = OrderedDict()
        output_args = OrderedDict()
        nonseq_args = OrderedDict()
        targets = OrderedDict()
        nNone = 0

        # pop cost function
        try:
            cost_obj = self.nodes.pop('cost')
            sorted_nodes.remove('cost')
        except KeyError:
            cost_obj = None

        for name, node in self.nodes.items():
            if hasattr(node, 'isroot'):
                if node.isroot:
                    seq_args[name] = node
                    seqs.append(node.out)
            if hasattr(node, 'istarget'):
                if node.istarget:
                    targets[name] = node
            if hasattr(node, 'get_init_state'):
                output_args[name] = node
                state = node.get_init_state()
                outputs.append(state)
        nstate = len(outputs)
        
        if given_args is not None:
            n_steps = given_args.pop('n_steps', None)
            reverse = given_args.pop('reverse', False)
        if cost_obj is not None:
            output_args['cost'] = cost_obj

        if given_output_args is not None:
            for name, arg in given_output_args.items():
                output_args[name] = arg
        nNone = len(output_args) - nstate
        if cost_obj is not None:
            nNone += 1
        outputs = flatten(outputs + [None] * nNone)

        if given_nonseq_args is not None:
            for name, arg in given_nonseq_args.items():
                nonseq_args[name] = arg

        def scan_fn(*args):
            ipdb.set_trace()
            while sorted_nodes:
                node = sorted_nodes.popleft()
                if self.nodes[node].isroot:
                    continue
                parent = self.nodes[node].parent
                inp = []
                for par in parent:
                    inp.append(par.out)
                self.nodes[node].out = self.nodes[node].fprop(inp)

        ipdb.set_trace()
        result, updates = theano.scan(
            fn=scan_fn,
            sequences=seqs,
            outputs_info=outputs,
            non_sequences=nonseqs,
            n_steps=n_steps,
            go_backwards=reverse)
        result = tolist(result)

        # Post scan, add cost
        inp = flatten(list([result, targets]))
        if cost_obj is not None:
            cost = cost_obj(inp)

        ipdb.set_trace()
        return cost

    def get_params(self):
        #return flatten([node.get_params() for node in self.nodes.values()])
        return flatten([node.get_params().values()
                        for node in self.nodes.values()])

    def get_inputs(self):
        return self.inputs
