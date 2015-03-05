import ipdb
import numpy as np


class Data(object):
    """
    Abstract class for data

    Parameters
    ----------
    .. todo::
    """
    def __init__(self):
        raise NotImplementedError(
            str(type(self)) + " does not implement Data.init.")

    def load_data(self, path):
        raise NotImplementedError(
            str(type(self)) + " does not implement Data.load_data.")

    def batch(self, i):
        raise NotImplementedError(
            str(type(self)) + " does not implement Data.batch.")

    def num_examples(self):
        raise NotImplementedError(
            str(type(self)) + " does not implement Data.num_examples.")

    def theano_vars(self):
        raise NotImplementedError(
            str(type(self)) + " does not implement Data.theano_vars.")


class DesignMatrix(Data):
    """
    Abstract class for static data.

    Parameters
    ----------
    .. todo::
    """
    def batch(self, data, i):
        batch = data[i*self.batchsize:(i+1)*self.batchsize]
        return batch

    def __iter__(self):
        return self

    def next(self):
        self.index += 1
        if self.index < self.nbatch:
            return (self.batch(data, self.index) for data in self.data)
        else:
            self.index = -1
            raise StopIteration()


class TemporalSeries(Data):
    """
    Abstract class for temporal data.
    We use TemporalSeries when the data contains variable length
    seuences, otherwise, we use DesignMatrix.

    Parameters
    ----------
    .. todo::
    """
    def batch(self, data, i):
        batch = data[i*self.batchsize:(i+1)*self.batchsize]
        return batch.swapaxes(0, 1)

    def __iter__(self):
        return self

    def create_mask(self, batch):
        samples_len = [len(sample) for sample in batch]
        max_sample_len = max(samples_len)
        mask = np.zeros((max_sample_len, len(batch)),
                        dtype=batch.dtype)
        for i, sample_len in enumerate(samples_len):
            mask[:sample_len, i] = 1
        return mask

    def zero_pad(self, batch):
        max_sample_len = max(len(sample) for sample in batch)
        rval = np.zeros((len(batch), max_sample_len,
                         batch[0].shape[-1]), batch.dtype)
        for i, sample in enumerate(batch):
            rval[i, :len(sample)] = sample
        return rval.swapaxes(0, 1)
