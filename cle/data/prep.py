import ipdb
import numpy as np

from scipy.fftpack import rfft


class SequentialPrepMixin(object):
    """
    Preprocessing mixin for sequential data

    Parameters
    ----------
    X : list of lists or ndArrays
    """
    def normalize_by_norm(self, X, mean_norm=None):
        if mean_norm is None:
            mean_norm = 0
            for i in range(len(X)):
                euclidean_norm = np.sqrt(np.square(X[i].sum()))
                X[i] /= euclidean_norm
                mean_norm += euclidean_norm
            mean_norm /= len(X)
        else:
            X = [x[i] / mean_norm for x in X]
        return X, mean_norm

    def normalize_by_global(self, X, X_mean=None, X_std=None):
        if (X_mean or X_std) is None:
            X_len = np.array([len(x) for x in X]).sum()
            X_mean = np.array([x.sum() for x in X]).sum() / X_len
            X_sqr = np.array([(x**2).sum() for x in X]).sum() / X_len
            X_std = np.sqrt(X_sqr - X_mean**2)
            X = (X - X_mean) / X_std
        else:
            X = (X - X_mean) / X_std
        return (X, X_mean, X_std)

    def standardize(self, X, X_max=None, X_min=None):
        if (X_max or X_min) is None:
            X_max = np.array([x.max() for x in X]).max()
            X_min = np.array([x.min() for x in X]).min()
            X = (X - X_min) / (X_max - X_min)
        else:
            X = (X - X_min) / (X_max - X_min)
        return (X, X_max, X_min)
