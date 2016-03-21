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
        self.nlabel = nlabel
        super(Music, self).__init__(**kwargs)

    def slices(self, start, end):
        batches = [mat[start:end] for mat in self.data]
        #mask = tolist(self.create_mask(batches[0]).swapaxes(0, 1))
        #batches = [self.zero_pad(batch) for batch in batches]
        x, x_mask = self.create_mask_and_zero_pad(batches[0])
        y, y_mask = self.create_mask_and_zero_pad(batches[1])
        #return totuple(batches + mask)
        return totuple([x, y, y_mask])

    def load(self, path):
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

    def theano_vars(self):
        return [T.ftensor3('x'), T.ftensor3('y'), T.fmatrix('mask')]

    def list2nparray(self, x, dim):
        z = np.zeros((dim,), dtype=np.float32)
        for i in x:
            z[i-1] = 1
        return z
