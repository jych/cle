import ipdb
import numpy as np

from cle.cle.layers import InitCell, StemCell
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.layers.recurrent import RecurrentLayer


class ReadLayer(RecurrentLayer):
    """
    Draw read layer

    Parameters
    ----------
    h_dec   : Linear
    x       : TensorVariable
    \hat{x} : Transformed TensorVariable
    """
    def __init__(self,
                 N,
                 init_U=InitCell('randn'),
                 **kwargs):
        super(ReadLayer, self).__init__(init_U, **kwargs)

    def fprop(self, XH):
        X, H = XH
        x = X[0]
        x_hat = X[1]
        h_dec = H[0]

    def filter_bank(self):



class WriteLayer(StemCell):
    """
    Draw write layer

    Parameters
    ----------
    .. todo::
    """





