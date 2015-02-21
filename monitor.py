import theano


class Monitor(object):
    """
    Monitoring class

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, graph):
        self.graph = graph

    def get_params(self):
        return self.graph.get_params()

    def fprop(self, x=None):
        rlist = []
        for node in self.sorted_nodes:
            rlist.append(node.out)
        return rlist
