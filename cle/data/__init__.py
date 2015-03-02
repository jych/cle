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
    Abstract class for data

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
