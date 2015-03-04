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
    """
    Abstract class for static data.

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, name, path, start=0, end=None, batchsize=None):
        self.name = name
        self.path = path
        self.batchsize = batchsize
        data = self.load_data(path)
        end = min(mat.shape[0] for mat in data) if end is None else end
        # TODO : verify start and end
        self.data = [mat[start:end] for mat in data]
        self.ndata = end - start
        self.batchsize = self.ndata if batchsize is None else batchsize
        self.nbatch = int(np.float(self.ndata / float(self.batchsize)))
        
    def load_data(self, path):
        raise NotImplementedError(
            str(type(self)) + " does not implement DesignMatrix.load_data.")


class TemporalSeries(DesignMatrix):
    """
    Abstract class for temporal data.
    We use TemporalSeries when the data contains variable length
    seuences, otherwise, we use DesignMatrix.

    Parameters
    ----------
    .. todo::
    """
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
