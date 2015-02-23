import theano.tensor as T
from data import *


class BouncingBalls(DesignMatrix):

    def theano_vars(self):
        return [T.fmatrix('x'), T.fmatrix('y')]
