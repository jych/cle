import ipdb
import theano.tensor as T

from cle.cle.layers import StemCell
from itertools import izip


class FullyConnectedLayer(StemCell):
    """
    Fully connected layer

    Parameters
    ----------
    .. todo::
    """
    """
    def __init__(self,
                 unit,
                 **kwargs):
        super(FullyConnectedLayer, self).__init__(**kwargs)
        self.nonlin = self.which_nonlin(unit)
    """
    def fprop(self, X):
        # X could be a list of inputs.
        # depending the number of parents.
        z = T.zeros(self.nout)
        for x, parent in izip(X, self.parent):
            W = self.params['W_'+parent.name+self.name]
            z += T.dot(x[:, :parent.nout], W)
        z += self.params['b_'+self.name]
        z = self.nonlin(z)
        z.name = self.name
        return z
