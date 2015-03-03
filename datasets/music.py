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
    def __init__(self, nlabel, **kwargs):
        super(Music, self).__init__(**kwargs)
        self.nlabel = nlabel
        self.data = self.load_data()
        self.ndata = self.num_examples()
        if self.batchsize is None:
            self.batchsize = self.ndata
        self.nbatch = int(np.float(self.ndata / float(self.batchsize)))
        self.index = -1

    def load_data(self):
        data = np.load(self.path)
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

    def num_examples(self):
        return self.data[0].shape[0]

    def batch(self, data, i):
        batch = data[i*self.batchsize:(i+1)*self.batchsize]
        return batch.swapaxes(0, 1)

    def __iter__(self):
        return self

    def next(self):
        self.index += 1
        if self.index < self.nbatch:
            batch = [self.batch(data, self.index) for data in self.data]
            mask = tolist(self.create_mask(batch[0].swapaxes(0, 1)))
            batch = [self.zero_pad(data) for data in batch]
            return totuple(batch + mask)
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
