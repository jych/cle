import theano.tensor as T
from cle.cle.data import *


class BouncingBalls(DesignMatrix):

    def theano_vars(self):
        return [T.ftensor('x'), T.ftensor('y')]
