import ipdb
import numpy as np


class Data(object):
    """
    Abstract class for data

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, name, path):
        self.name = name
        self.data = self.load(path)

    def load(self, path):
        return np.load(path)

    def slices(self, i):
        raise NotImplementedError(
            str(type(self)) + " does not implement Data.slice.")

    def num_examples(self):
        return max(mat.shape[0] for mat in self.data)

    def theano_vars(self):
        raise NotImplementedError(
            str(type(self)) + " does not implement Data.theano_vars.")


class Iterator(object):
    """
    Dataset iterator

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, data, batch_size=None, nbatch=None,
                 start=0, end=None):
        if (batch_size or nbatch) is None:
            raise ValueError("Either batch_size or nbatch should be given.")
        if (batch_size and nbatch) is not None:
            raise ValueError("Provide either batch_size or nbatch.")
        self.start = start
        self.end = data.num_examples() if end is None else end
        if self.start >= self.end or self.start < 0:
            raise ValueError("Got wrong value for start %d.", self.start)
        self.nexp = self.end - self.start
        if nbatch is not None:
            self.batch_size = int(np.float(self.nexp / float(nbatch)))
            self.nbatch = nbatch
        elif batch_size is not None:
            self.batch_size = batch_size
            self.nbatch = int(np.float(self.nexp / float(batch_size)))
        self.data = data
        self.name = self.data.name

    def __iter__(self):
        start = self.start
        end = self.end - self.end % self.batch_size
        for idx in xrange(start, end, self.batch_size):
            yield self.data.slices(idx, idx + self.batch_size)

class DesignMatrix(Data):
    """
    Abstract class for static data.

    Parameters
    ----------
    .. todo::
    """
    def slices(self, start, end):
        return (mat[start:end] for mat in self.data)


class TemporalSeries(Data):
    """
    Abstract class for temporal data.
    We use TemporalSeries when the data contains variable length
    seuences, otherwise, we use DesignMatrix.

    Parameters
    ----------
    .. todo::
    """
    def slices(self, start, end):
        return (mat[start:end].swapaxes(0, 1)
                for mat in self.data)

    def create_mask(self, batch):
        samples_len = [len(sample) for sample in batch]
        max_sample_len = max(samples_len)
        mask = np.zeros((max_sample_len, len(batch)),
                        dtype=batch.dtype)
        for i, sample_len in enumerate(samples_len):
            mask[:sample_len, i] = 1.
        return mask, max_sample_len

    def zero_pad(self, batch, max_sample_len):
        rval = np.zeros((len(batch), max_sample_len,
                         batch[0].shape[-1]), batch.dtype)
        for i, sample in enumerate(batch):
            rval[i, :len(sample)] = sample
        return rval.swapaxes(0, 1)
