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
    def __init__(self, name, path, batchsize=None):
        self.name = name
        self.path = path
        self.batchsize = batchsize

    def load_data(self):
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
        samples_length =\
            [len(sample.flatten().nonzero()[0]) if sample[0] != 0
             else len(sample.flatten().nonzero()[0]) + 1 for sample in batch]
        mask = np.zeros((max(samples_length), len(batch)), dtype=config.floatX)
        for i, sample_length in enumerate(samples_length):
            mask[:sample_length, i] = 1
        return mask
