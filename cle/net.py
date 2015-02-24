import numpy as np

from cle.cle.util import *
from cle.cle.layers import *
from cle.cle.layers.layer import *
from collections import deque


def topological_sort(graph):
    """
    Topological sort

    Parameters
    ----------
    None
    """
    GRAY, BLACK = 0, 1
    order, enter, state = deque(), set(graph), {}
    this_graph = dict()
    for node in graph:
        this_graph[node] = tolist(graph[node])
    def dfs(node):
        state[node] = GRAY
        for k in this_graph.get(node, ()):
            sk = state.get(k, None)
            if sk == GRAY:
                raise ValueError("cycle")
            if sk == BLACK:
                continue
            enter.discard(k)
            dfs(k)
        order.appendleft(node)
        state[node] = BLACK
    while enter:
        dfs(enter.pop())
    return order


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
        self.seg_graph(nodes)
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
            if isinstance(self.nodes[node], InputLayer):
                continue 
            parent = self.nodes[node].parent
            inp = []
            for par in parent:
                inp.append(par.out)
            self.nodes[node].out = self.nodes[node].fprop(inp)
    
    def get_params(self):
        return flatten([node.get_params() for node in self.nodes.values()])

    def get_inputs(self):
        return self.inputs
