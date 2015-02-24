import theano.tensor as T
from cle.cle.data import *


class MNIST(DesignMatrix):

    def theano_vars(self):
        return [T.fmatrix('x'), T.lvector('y')]
