import numpy as np

from util import *
from layer import *
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


class Net(Layer):
    """
    Abstract class for networks

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, graph):
        self.graph = graph
        self.set_nodes()
        self.build_map()
        self.set_edges()
        self.build_parent_map(graph)
        self.params = self.get_params()

    def set_nodes(self):
        nodes = uniqify(flatten(self.graph))
        self.nodes = {}
        for node in nodes:
            self.nodes[node.name] = node

    def set_edges(self):
        self.edges = {}
        for pairs in self.graph:
            pairs[0] = tolist(pairs[0])
            for node in pairs[0]:
                self.edges[node.name] = pairs[1].name

    def backtrace(self, idx):
        return np.where(self.parent_map[:, idx]==1)[0]

    def build_graph(self):
        sorted_nodes = topological_sort(self.edges)
        while sorted_nodes:
            nodeid = sorted_nodes.popleft()
            if isinstance(self.nodes[nodeid], Input):
                continue 
            parent = self.backtrace(self.sym2idx[nodeid])
            if len(parent) > 1:
                inp = []
                for idx in parent:
                    parentid = self.idx2sym[unpack(idx)]
                    parent_node = self.nodes[parentid]
                    inp.append(parent_node.out)
            else:
                parentid = self.idx2sym[unpack(parent)]
                parent_node = self.nodes[parentid]
                inp = parent_node.out
            self.nodes[nodeid].out = self.nodes[nodeid].fprop(inp)
    
    def build_map(self):
        nodes = uniqify(flatten(self.graph))
        names = [node.name for node in nodes]
        inputs = [[edge.name for edge in tolist(edges[0])]
                  for edges in self.graph]
        outputs = [[edge.name for edge in tolist(edges[1])]
                  for edges in self.graph]
        seen = []
        for inps in inputs:
            unseen_tok = 0
            for inp in inps:
                if inp not in seen:
                    seen.append(inp)
                    unseen_tok = 1
                    unseen = inp
                else:
                    if unseen_tok:
                        a, b = seen.index(inp), seen.index(unseen)
                        seen[b], seen[a] = seen[a], seen[b]
        for outs in outputs:
            for out in outs:
                if out not in seen:
                    seen.append(out)
        sym2idx = OrderedDict()
        idx2sym = OrderedDict()
        for i, node in enumerate(seen):
            sym2idx[node] = i
            idx2sym[i] = node
        self.sym2idx = sym2idx
        self.idx2sym = idx2sym

    def build_parent_map(self, edges):
        n_node = len(self.nodes)
        parent_map = np.zeros((n_node, n_node))
        for pair in edges:
            parents, child = pair
            for parent in tolist(parents):
                parent_map[self.sym2idx[parent.name],
                           self.sym2idx[child.name]] = 1
        self.parent_map = parent_map

    def get_inputs(self):
        inputs = [node.out for node in self.nodes.values()
                  if node.__class__ is Input]
        inputs.sort(key=lambda x:x.name)
        return inputs

    def get_params(self):
        return flatten([node.get_params() for node in self.nodes.values()])
