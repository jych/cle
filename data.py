import ipdb
import numpy as np
import theano.tensor as T


class Data(object) :
    """
    Abstract class for data

    Parameters
    ----------
    .. todo::
    """
    def __init__(self):
        raise NotImplementedError(
            str(type(self)) + " does not implement Data.init.")

    def num_examples(self):
        raise NotImplementedError(
            str(type(self)) + " does not implement Data.num_examples.")

    def batch(self, i):
        raise NotImplementedError(
            str(type(self)) + " does not implement Data.batch.")

    def theano_vars(self):
        raise NotImplementedError(
            str(type(self)) + " does not implement Data.theano_vars.")


class DesignMatrix(Data):

    def __init__(self, data, which_set, batch_size):
        self.data = data[which_set]
        self.ndata = self.num_examples()
        self.batch_size = batch_size if batch_size is not None else self.ndata
        self.nbatch = self.ndata / self.batch_size + 1
        self.index = -1

    def num_examples(self): 
        return self.data[0].shape[0]

    def batch(self, data, i, size):
        return data[i*size:(i+1)*size]

    def __iter__(self):
        return self

    def next(self):
        self.index += 1
        if self.index < self.nbatch:
            return (self.batch(data, self.index, self.batch_size)
                    for data in self.data)
        else:
            self.index = -1
            raise StopIteration()
