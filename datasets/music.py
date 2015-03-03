import ipdb
import numpy as np
import theano.tensor as T
from cle.cle.data import TemporalSeries


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
        self.nbatch = int(np.ceil(self.ndata / float(self.batchsize)))
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
            [np.asarray([self.map2array(ts, self.nlabel)
             for ts in np.asarray(d[:-1])]) for d in data])
        y = np.asarray(
            [np.asarray([self.map2array(ts, self.nlabel)
             for ts in np.asarray(d[1:])]) for d in data])
        return (X, y)

    def num_examples(self):
        return self.data[0].shape[0]

    def batch(self, data, i):
        size = self.batchsize
        ndata = self.ndata
        this_batch = data[i*size:min((i+1)*size, ndata)]
        if self.batch_size > 1:
            this_mask = self.create_mask(this_batch)
            return (this_batch.swapaxes(0, 1), this_mask.swapaxes(0, 1))
        return this_batch.swapaxes(0, 1)

    def __iter__(self):
        return self

    def next(self):
        self.index += 1
        if self.index < self.nbatch:
            batch = (self.batch(data, self.index) for data in self.data)
            if self.batch_size > 1:
                this_mask = self.create_mask(batch[0].swapaxes(0, 1))
                return batch + (this_mask.swapaxes(0, 1)) 
            return batch
        else:
            self.index = -1
            raise StopIteration()

    def theano_vars(self):
        return [T.ftensor3('x'), T.ftensor3('y'), T.ftensor3('mask')]

    def map2array(self, x, dim):
        z = np.zeros((dim,), dtype=np.float32)
        for i in x:
            z[i-1] = 1
        return z
