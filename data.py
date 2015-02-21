import numpy as np


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

    def num_data(self):
        raise NotImplementedError(
            str(type(self)) + " does not implement Data.num_data.")

    def batch(self, i):
        raise NotImplementedError(
            str(type(self)) + " does not implement Data.batch.")


class DesignMatrix(Data):

    def __init__(self, numpy_data):
        self.data = numpy_data

    def num_data(self): 
        return self.data.shape[0]

    def batch(self, i, size):
        return self.data[i*size: (i+1)*size]


class BatchProvider(object):
    def __init__(self, data_list, batch_size=None):
        self.data_list = data_list
        self.n_data = min([data.num_data() for data in data_list])
        self.batch_size = batch_size if batch_size is not None else self.n_data
        self.n_batches = self.n_data / self.batch_size + 1
        self.index = -1

    def __iter__(self):
        return self

    def next(self):
        self.index += 1
        if self.index < self.n_batches:
            return (data.batch(self.index, self.batch_size)
                    for data in self.data_list)
        else:
            self.index = -1
            raise StopIteration()
