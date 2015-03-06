import ipdb
import numpy as np

from scipy.fftpack import rfft


class SequentialPrepMixin(object):
    """
    Preprocessing mixin for sequential data
    """
    def norm_normalize(self, X, avr_norm=None):
        """
        Unify the norm of each sequence in X

        Parameters
        ----------
        X       : list of lists or ndArrays
        avr_nom : Scalar
        """
        if avr_norm is None:
            avr_norm = 0
            for i in range(len(X)):
                euclidean_norm = np.sqrt(np.square(X[i].sum()))
                X[i] /= euclidean_norm
                avr_norm += euclidean_norm
            avr_norm /= len(X)
        else:
            X = [x[i] / avr_norm for x in X]
        return X, avr_norm

    def global_normalize(self, X, X_mean=None, X_std=None):
        """
        Globally normalize X into zero mean and unit variance

        Parameters
        ----------
        X      : list of lists or ndArrays
        X_mean : Scalar
        X_std  : Scalar

        Notes
        -----
        Compute varaince using the relation
        >>> Var(X) = E[X^2] - E[X]^2
        """   
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
        """
        Standardize X such that X \in [0, 1]

        Parameters
        ----------
        X     : list of lists or ndArrays
        X_max : Scalar
        X_min : Scalar
        """
        if (X_max or X_min) is None:
            X_max = np.array([x.max() for x in X]).max()
            X_min = np.array([x.min() for x in X]).min()
            X = (X - X_min) / (X_max - X_min)
        else:
            X = (X - X_min) / (X_max - X_min)
        return (X, X_max, X_min)
