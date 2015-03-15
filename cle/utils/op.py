import ipdb
import numpy as np
import theano.tensor as T


def dropout(x, p, theano_rng):
    if p < 0 or p > 1:
        raise ValueError("p should be in [0, 1].")
    mask = theano_rng.binomial(p=p, size=x.shape,
                               dtype=x.dtype)
    return x * mask


def logsumexp(x, axis=None):
    x_max = T.max(x, axis=axis, keepdims=True)
    z = T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return z.sum(axis=axis)


def add_noise(x, stddev, theano_rng):
    x += theano_rng.normal(size=x.shape,
                           avg=0.,
                           std=stddev,
                           dtype=x.dtype)
    return x


def overlap_sum(X, overlap):
    """
    WRITEME

    Parameters
    ----------
    x       : list of lists or ndArrays
    overlap : amount of overlap (usually half of the window size)

    Notes
    -----
    This function assumes x as 3D
    """
    new_X = []
    for i in xrange(len(X)):
        len_x = len(X[i][0])
        time_steps = len(X[i])
        new_x = np.zeros(len_x + (time_steps - 1) * overlap)
        start = 0
        for j in xrange(time_steps):
            new_x[start:start+len_x] += X[i][j]
            start += overlap
        new_X.append(new_x[:-1])
    return np.asarray(new_X)
