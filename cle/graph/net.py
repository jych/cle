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

    def build_recurrent_graph(self, **kwargs):
        self.given_nonseq_args = kwargs.pop('nonseq_args', None)
        self.given_output_args = kwargs.pop('output_args', None)
        self.given_args = kwargs.pop('given_args', None)
        n_steps = None
        reverse = None
        seqs = []
        outputs = []
        nonseqs = []
        self.seq_args = OrderedDict()
        self.output_args = OrderedDict()

        for name, node in self.nodes.items():
            if hasattr(node, 'isroot'):
                if node.isroot:
                    self.seq_args[name] = node
                    seqs.append(node.out)
            if hasattr(node, 'get_init_state'):
                self.output_args[name] = node
                state = node.get_init_state()
                outputs.append(state)

        def scan_fn(*args):
            next_recurrence = []
            sorted_nodes = topological_sort(self.graph)
            inputs = tolist(args[:len(self.seq_args)])
            recurrence = tolist(args[len(self.seq_args):
                                     len(self.seq_args)+
                                     len(self.output_args)])
            for nname, node in self.nodes.items():
                for i, (aname, arg) in enumerate(self.seq_args.items()):
                    if nname == aname:
                        node.out = inputs[i]
                for i, (aname, arg) in enumerate(self.output_args.items()):
                    if nname == aname:
                        node.rec_out = recurrence[i]
            while sorted_nodes:
                node = sorted_nodes.popleft()
                if self.nodes[node].isroot:
                    continue
                parent = self.nodes[node].parent
                inp = []
                for par in parent:
                    inp.append(par.out)
                if hasattr(self.nodes[node], 'recurrent'):
                    recurrent = self.nodes[node].recurrent
                    rec_inp = []
                    for rec in recurrent:
                        rec_inp.append(rec.rec_out)
                    inp = [inp, rec_inp]
                    self.nodes[node].out = self.nodes[node].fprop(inp)
                    next_recurrence.append(self.nodes[node].out)
                    continue
                self.nodes[node].out = self.nodes[node].fprop(inp)
            required_outputs = []
            for node in self.nodes.values():
                if isinstance(self.given_output_args, dict):
                    for arg in self.given_output_args.values():
                        if node is arg:
                            required_outputs.append(node.out)
                elif isinstance(self.given_output_args, list):
                    for arg in self.given_output_args:
                        if node is arg:
                            required_outputs.append(node.out)
            self.nNone = len(required_outputs)
            return next_recurrence + required_outputs

        dummy_args = seqs + outputs + nonseqs
        dummy = scan_fn(*dummy_args)
        outputs = flatten(outputs + [None] * self.nNone)

        if self.given_args is not None:
            n_steps = self.given_args.pop('n_steps', None)
            reverse = self.given_args.pop('reverse', False)
        if self.given_nonseq_args is not None:
            if isinstance(self.given_nonseq_args, dict):
                for arg in self.given_nonseq_args.values():
                    nonseqs.append(arg)
            elif isinstance(self.given_nonseq_args, list):
                for arg in self.given_nonseq_args:
                    nonseqs.append(arg)

        result, updates = theano.scan(
            fn=scan_fn,
            sequences=seqs,
            outputs_info=outputs,
            non_sequences=nonseqs,
            n_steps=n_steps,
            go_backwards=reverse)
        result = tolist(result)

        return result[len(self.seq_args):]

    def get_params(self):
        return flatten([node.get_params().values()
                        for node in self.nodes.values()])

    def get_inputs(self):
        return self.inputs
