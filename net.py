import numpy as np

from util import *
from layer import *
from collections import deque


def build_parent_matrix(nodes, edges, sym2idx):
    """
    Mapping generator between nodes and indices

    Parameters
    ----------
    .. todo::
    """
    parent_map = np.zeros((len(nodes), len(nodes)))
    for node in edges:
        parent_map[sym2idx[node], sym2idx[edges[node]]] = 1
    return parent_map


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


class Net(Layer):
    """
    Abstract class for networks

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.build_map(self.nodes)
        self.params = self.get_params()

    def get_inputs(self):
        inputs = [node.out for node in self.nodes.values()
                  if node.__class__ is Input]
        inputs.sort(key=lambda x:x.name)
        return inputs

    def get_params(self):
        return flatten([node.get_params() for node in self.nodes.values()])

    def build_graph(self):
        self.parent_map =\
            build_parent_matrix(self.nodes, self.edges, self.sym2idx)
        sorted_nodes = topological_sort(self.edges)
        self.sorted_nodes = sorted_nodes
        while sorted_nodes:
            node = sorted_nodes.popleft()
            if isinstance(self.nodes[node], Input):
                continue 
            parent = self.backtrace(self.sym2idx[node])
            if len(parent) > 1:
                inp = []
                for idx in parent:
                    parent_node = self.idx2sym[idx]
                    inp.append(self.nodes[parent_node].out)
            else:
                parent_node = self.idx2sym[parent[0]]
                inp = self.nodes[parent_node].out
            self.nodes[node].out = self.nodes[node].fprop(inp)

    def backtrace(self, idx):
        parent = np.where(self.parent_map[:, idx]==1)[0]
        return parent

    def build_map(self, syms):
        self.sym2idx = {sym:i for i, sym in enumerate(syms)}
        self.idx2sym = {i:sym for i, sym in enumerate(syms)}

    def add_node(self, nodes):
        for key, val in nodes.items():
            self.nodes[key] = val
        self.build_map(self.nodes)
        self.params = self.get_params()

    def add_edge(self, edge):
        for key, val in edge.items():
            self.edges[key] = val
