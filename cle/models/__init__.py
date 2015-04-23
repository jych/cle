import ipdb
import theano

from collections import OrderedDict
from cle.cle.utils import (
    flatten,
    tolist,
    topological_sort,
    PickleMixin
)


class Model(object):
    """
    Abstract class for models

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, inputs=None, graphs=None, nodes=None, params=None, updates=None):
        self.inputs = inputs
        self.graphs = graphs
        self._params = params
        # used in manual mode
        self.nodes = nodes
        self.updates = OrderedDict()
        if updates is not None:
            for update in updates:
                self.updates[update] = update

    @property
    def params(self):
        if getattr(self, '_params', None) is None:
            self._params = self.get_params()
        return self._params

    def get_params(self):
        params = []
        for graph in tolist(self.graphs):
            params += graph.params
        return params

    def set_updates(self, updates):
        for update in updates:
            self.updates[update] = update
