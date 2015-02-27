import theano.tensor as T
from cle.cle.data import *


class BouncingBalls(DesignMatrix):
    def __init__(self, name, data, batch_size=None):
        self.name = name
        self.data = data
        self.ndata = self.num_examples()
        self.batch_size = batch_size if batch_size is not None else self.ndata
        self.nbatch = int(np.ceil(self.ndata / float(self.batch_size)))
        self.index = -1

    def theano_vars(self):
        return [T.ftensor('x'), T.ftensor('y')]
