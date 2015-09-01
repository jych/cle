import ipdb
import theano

from cle.cle.utils import (
    flatten,
    tolist,
    topological_sort,
    PickleMixin
)

from collections import OrderedDict


class Model(object):
    """
    Abstract class for models

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, inputs=None, nodes=None, params=None, updates=None):
        self.inputs = inputs
        self.nodes = nodes
        self.params = params
        self.updates = OrderedDict()
        if updates is not None:
            for update in updates.items():
                self.updates[update[0]] = update[1]

    def set_updates(self, updates):

        for update in updates.items():
            self.updates[update[0]] = update[1]
