import cPickle, gzip, os, sys, time, re, ipdb
import numpy as np

from theano.compat.python2x import OrderedDict
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from util import *
from layer import *
from collections import deque


def linked_list_to_sparse_matrix(nodes, edges, sym_to_idx):
    sparse_matrix = np.zeros((len(nodes), len(nodes)))
    for node in edges:
        sparse_matrix[sym_to_idx[node], sym_to_idx[edges[node]]] = 1
    return sparse_matrix


def topological_sort(graph):
    GRAY, BLACK = 0, 1
    order, enter, state = deque(), set(graph), {}
    for node in graph:
        if graph.get(node, ()) is not list:
            graph[node] = [graph[node]]

    def dfs(node):
        state[node] = GRAY
        for k in graph.get(node, ()):
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
    todo..
    """
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.build_graph()

    def get_params(self):
        return flatten([node.get_params() for node in self.nodes])

    def fprop(self, x):

        return x

    def build_graph(self):
        sym_to_idx = {node:i for i, node in enumerate(self.nodes)}
        idx_to_sym = {i:node for i, node in enumerate(self.nodes)}
        sparse_matrix =\
            linked_list_to_sparse_matrix(self.nodes, self.edges, sym_to_idx)
        sorted_edges = topological_sort(self.edges)
        while sorted_edges:
            node = sorted_edges.popleft()
            if isinstance(self.nodes[node], Input):
                continue
            parent = np.where(sparse_matrix[:, sym_to_idx[node]]==1)[0]
            if len(parent) > 1:
                inp = []
                for idx in parent:
                    parent_node = idx_to_sym[idx]
                    inp.append(self.nodes[parent_node].out)
            else:
                parent_node = idx_to_sym[parent[0]]
                inp = self.nodes[parent_node].out
            self.nodes[node].out = self.nodes[node].fprop(inp)

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)
