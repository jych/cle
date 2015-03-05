import ipdb
import numpy as np
import theano
import theano.tensor as T

from collections import OrderedDict
from cle.cle.utils import (
    flatten,
    tolist,
    topological_sort,
    PickleMixin
)


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
        #self.set_graph(nodes)
        self.set_nodes(nodes)
        self.set_graph()
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

    def set_graph(self):
        self.graph = {}
        for nname, node in self.nodes.items():
            if not node.isroot:
                parent = node.parent
                for par_node in tolist(parent):
                    if par_node.name in self.graph.keys():
                        self.graph[par_node.name] =\
                            tolist(self.graph[par_node.name]) +\
                            [node.name]
                    else:
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

    def build_recurrent_graph(self, n_steps=None, reverse=False, **kwargs):
        self.nonseq_args = kwargs.pop('nonseq_args', None)
        self.output_args = kwargs.pop('output_args', None)
        self.context_args = kwargs.pop('context_args', None)
        self.iterators = kwargs.pop('iterators', None)
        seqs = []
        inputs = []
        outputs = []
        nonseqs = []
        self.input_args = OrderedDict()
        self.recur_args = OrderedDict()
        for name, node in self.nodes.items():
            if hasattr(node, 'isroot'):
                if node.isroot:
                    self.input_args[name] = node
                    inputs.append(node.out)
            if hasattr(node, 'get_init_state'):
                self.recur_args[name] = node
                state = node.get_init_state()
                outputs.append(state)
        self.nrecur = len(self.recur_args)
        # Substitutes initial hidden state into a context
        if self.context_args is not None:
            for i, (nname, node) in enumerate(self.output_args.items()):
                for aname, arg in self.context_args.items():
                    if nname == aname:
                        outputs[i] = arg
        if self.iterators is None:
            seqs += inputs
        elif self.iterators is not None:
            seqs += inputs[len(self.iterators):]
            outputs += inputs[:len(self.iterators)]
        if self.output_args is not None:
            self.nNone = len(self.output_args)
        else:
            self.nNone = 0
        outputs = flatten(outputs + [None] * self.nNone)
        if self.nonseq_args is not None:
            for arg in self.nonseq_args:
                nonseqs.append(arg)
        self.nseqs = len(seqs)
        self.noutputs = len(outputs)
        self.nnonseqs = len(nonseqs)
        result, updates = theano.scan(
            fn=self.scan_fn,
            sequences=seqs,
            outputs_info=outputs,
            non_sequences=nonseqs,
            n_steps=n_steps,
            go_backwards=reverse)
        result = tolist(result)
        for k, v in updates.iteritems():
            k.default_update = v
        return result[self.nrecur:]

    def scan_fn(self, *args):
        next_recurrence = []
        sorted_nodes = topological_sort(self.graph)
        inputs = tolist(args[:self.nseqs])
        recurrence = tolist(args[self.nseqs:
                                 self.nseqs+
                                 self.nrecur])
        inputs += tolist(args[self.nseqs+self.nrecur:
                              self.nseqs+self.noutputs])
        nonseqs = tolist(args[self.nseqs+
                              self.noutputs:])
        for nname, node in self.nodes.items():
            for i, (aname, arg) in enumerate(self.input_args.items()):
                if node is arg:
                    node.out = inputs[i]
            for i, (aname, arg) in enumerate(self.recur_args.items()):
                if node is arg:
                    node.rec_out = recurrence[i]
        while sorted_nodes:
            node = sorted_nodes.popleft()
            if self.nodes[node].isroot:
                continue
            parent = self.nodes[node].parent
            inp = []
            for par in parent:
                inp.append(par.out)
            if self.nodes[node] in self.recur_args.values():
                recurrent = self.nodes[node].recurrent
                rec_inp = []
                for rec in recurrent:
                    rec_inp.append(rec.rec_out)
                inp = [inp, rec_inp]
                self.nodes[node].out = self.nodes[node].fprop(inp)
                next_recurrence.append(self.nodes[node].out)
            else:
                self.nodes[node].out = self.nodes[node].fprop(inp)
        required_outputs = []
        if self.iterators is not None:
            for arg in self.iterators:
                for node in self.nodes.values():
                    if node is arg:
                        required_outputs.append(node.out)
        if self.output_args is not None:
            for arg in self.output_args:
                for node in self.nodes.values():
                    if node is arg:
                        required_outputs.append(node.out)
        return next_recurrence + required_outputs

    def get_params(self):
        return flatten([node.get_params().values()
                        for node in self.nodes.values()])

    def get_inputs(self):
        return self.inputs

    def add_node(self, node):
        self.nodes[node.name] = node
        self.set_graph()

    def del_node(self, node):
        try:
            del self.nodes[node.name]
        except KeyError as e:
            print("There is no such node %s.", node.name)
        self.set_graph()
