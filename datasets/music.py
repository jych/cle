import ipdb
import numpy as np
import theano.tensor as T

from cle.cle.data import TemporalSeries
from cle.cle.utils import tolist, totuple


class Music(TemporalSeries):
    """
    Music datasets batch provider

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, name, path, nlabel, batchsize=None):
        self.name = name
        self.path = path
        self.batchsize = batchsize
        self.nlabel = nlabel
        self.data = self.load_data(path)
        self.nexp = self.num_examples()
        self.batchsize = self.nexp if batchsize is None else batchsize
        self.nbatch = int(np.float(self.nexp / float(self.batchsize)))
        self.index = -1

    def num_examples(self):
        return self.data[0].shape[0]

    def load_data(self, path):
        data = np.load(path)
        if self.name == 'train':
            data = data['train']
        elif self.name == 'valid':
            data = data['valid']
        elif self.name == 'test':
            data = data['test']
        X = np.asarray(
            [np.asarray([self.list2nparray(ts, self.nlabel)
             for ts in np.asarray(d[:-1])]) for d in data])
        y = np.asarray(
            [np.asarray([self.list2nparray(ts, self.nlabel)
             for ts in np.asarray(d[1:])]) for d in data])
        return (X, y)

    def next(self):
        self.index += 1
        if self.index < self.nbatch:
            batches = [self.batch(data, self.index) for data in self.data]
            mask = tolist(self.create_mask(batches[0].swapaxes(0, 1)))
            batches = [self.zero_pad(batch) for batch in batches]
            return totuple(batches + mask)
        else:
            self.index = -1
            raise StopIteration()

    def theano_vars(self):
        return [T.ftensor3('x'), T.ftensor3('y'), T.fmatrix('mask')]

    def list2nparray(self, x, dim):
        z = np.zeros((dim,), dtype=np.float32)
        for i in x:
            z[i-1] = 1
        return z
