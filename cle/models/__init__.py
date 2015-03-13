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
    def __init__(self, inputs=None, graphs=None):
        self.inputs = inputs
        self.graphs = graphs

    def get_params(self):
        params = []
        for graph in tolist(self.graphs):
            params += graph.params
        return params
