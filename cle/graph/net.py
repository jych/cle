import ipdb
import theano

from collections import OrderedDict
from cle.cle.utils import (
    flatten,
    todict,
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
    def __init__(self, nodes, inputs, inputs_dim):
        # inputs and inputs_dim is Dict or OrderedDict
        self.inputs = todict(inputs)
        self.inputs_dim = inputs_dim
        self.set_nodes(nodes)
        self.sorted_nodes = []
        self.set_graph()
        self.initialize()
        self.params = self.get_params()

    def set_nodes(self, nodes):
        self.nodes = {}
        for node in nodes:
            self.nodes[node.name] = node

    def initialize(self):
        for node in self.nodes.values():
            node.initialize()

    def set_batch_size(self, batch_size):
        for node in self.nodes.values():
            if hasattr(node, 'batch_size'):
                node.batch_size = batch_size

    def set_graph(self):
        self.graph = {}
        for nname, node in self.nodes.items():
            parent = node.parent
            for par in tolist(parent.keys()):
                if par in self.inputs.keys():
                    continue
                if par in self.graph.keys():
                    self.graph[par] =\
                        tolist(self.graph[par]) + [node.name]
                else:
                    self.graph[par] = node.name
        sorted_nodes = topological_sort(self.graph)
        if len(self.graph) > 0:
            for i in xrange(len(self.nodes)):
                self.sorted_nodes.append(sorted_nodes.popleft())
        for node in self.nodes:
            parent = self.nodes[node].parent
            for par in tolist(parent.keys()):
                try:
                    self.nodes[node].parent[par] = self.inputs_dim[par]
                except:
                    if self.nodes[par].nout is not None:
                        # Assume this is FullyConnectedLayer
                        self.nodes[node].parent[par] = self.nodes[par].nout
                    else:
                        # Assume this is ConvLayer
                        try:
                            self.nodes[node].parent[par] = self.nodes[par].outshape
                        except:
                            # Assume this is MaxPool2D
                            self.nodes[par].initialize()
                            self.nodes[node].parent[par] = self.nodes[par].outshape
            if hasattr(node, 'recurrent'):
                recurrent = self.nodes[node].recurrent
                for rec in tolist(recurrent.keys()):
                    self.nodes[node].recurrent[rec] = self.nodes[rec].nout

    def build_graph(self):
        for node in self.sorted_nodes:
            inp = []
            parent = self.nodes[node].parent
            for par in parent:
                try:
                    inp.append(self.inputs[par])
                except:
                    inp.append(self.nodes[par].out)
            self.nodes[node].out = self.nodes[node].fprop(inp)

    def build_recurrent_graph(self, n_steps=None, reverse=False, **kwargs):
        self.nonseq_args = kwargs.pop('nonseq_args', None)
        self.output_args = kwargs.pop('output_args', None)
        self.context_args = kwargs.pop('context_args', None)
        self.iterators = kwargs.pop('iterators', None)
        self.nonseq_inputs = kwargs.pop('nonseq_inputs', None)
        self.nNone = 0
        inputs = self.inputs.values()
        seqs = []
        outputs = []
        nonseqs = []
        if self.nonseq_inputs is not None:
            for i in self.nonseq_inputs:
                nonseqs.append(inputs.pop(i))
        self.input_args = OrderedDict()
        self.recur_args = OrderedDict()
        for name, node in self.nodes.items():
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
        if self.output_args is None and self.iterators is None:
            return result
        if len(updates) == 0:
            return result[-self.nNone:]
        for k, v in updates.iteritems():
            k.default_update = v
        return result[-self.nNone:], updates

    def scan_fn(self, *args):
        next_recurrence = []
        sorted_nodes = topological_sort(self.graph)
        inputs = tolist(args[:self.nseqs])
        recurrence = tolist(args[self.nseqs:self.nseqs+self.nrecur])
        inputs += tolist(args[self.nseqs+self.nrecur:self.nseqs+self.noutputs])
        nonseqs = tolist(args[self.nseqs+self.noutputs:])
        for nname, node in self.nodes.items():
            for i, (aname, arg) in enumerate(self.recur_args.items()):
                if node is arg:
                    node.rec_out = recurrence[i]
        if len(sorted_nodes) != 0:
            for node in self.sorted_nodes:
                inp = []
                parent = self.nodes[node].parent
                for par in parent:
                    tok = 1
                    for inp2 in inputs:
                        if par in inp2.name:
                            inp.append(inp2)
                            tok = 0
                            break
                    if tok:
                        inp.append(self.nodes[par].out)
                if self.nodes[node] in self.recur_args.values():
                    rec_inp = []
                    recurrent = self.nodes[node].recurrent
                    for rec in recurrent:
                        rec_inp.append(self.nodes[rec].rec_out)
                    inp = [inp, rec_inp]
                    self.nodes[node].out = self.nodes[node].fprop(inp)
                    next_recurrence.append(self.nodes[node].out)
                else:
                    self.nodes[node].out = self.nodes[node].fprop(inp)
        else:
            # Assume that you have only single depth (parallel) graph
            # Instead of Queue use for-loop to forcibly run the operation
            for node in self.nodes:
                inp = []
                parent = self.nodes[node].parent
                for par in parent:
                    tok = 1
                    for inp2 in inputs:
                        if par in inp2.name:
                            inp.append(inp2)
                            tok = 0
                            break
                    if tok:
                        inp.append(self.nodes[par].out)
                if self.nodes[node] in self.recur_args.values():
                    rec_inp = []
                    recurrent = self.nodes[node].recurrent
                    for rec in recurrent:
                        rec_inp.append(self.nodes[rec].rec_out)
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
        if len(self.inputs) != 0:
            return self.inputs.values()
        else:
            return []

    def add_input(self, inputs):
        for inp in inputs:
            self.inputs[inp.name] = inp

    def reset_input(self, inputs):
        self.inputs = OrderedDict()
        for inp in inputs:
            self.inputs[inp.name] = inp

    def add_node(self, nodes):
        for node in tolist(nodes):
            self.nodes[node.name] = node
        self.params = self.get_params()

    def del_node(self, node):
        if isinstance(node, str):
            try:
                del self.nodes[node]
            except KeyError:
                print("There is no such node %s.", node)
        else:
            try:
                del self.nodes[node.name]
            except KeyError:
                print("There is no such node %s.", node.name)
        # In case of removing nodes, you need to manually
        # set graph after deleting all the nodes.
        self.params = self.get_params()
