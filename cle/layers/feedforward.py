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
    def fprop(self, X):
        if len(X) != len(self.parent):
            raise AttributeError("The number of inputs doesn't match "
                                 "with the number of parents.")
        # X could be a list of inputs.
        # depending the number of parents.
        z = T.zeros((X[0].shape[0], self.nout))
        for x, (parname, parout) in izip(X, self.parent.items()):
            W = self.params['W_'+parname+'__'+self.name]
            z += T.dot(x[:, :parout], W)
        z += self.params['b_'+self.name]
        z = self.nonlin(z) + self.cons
        z.name = self.name
        return z
