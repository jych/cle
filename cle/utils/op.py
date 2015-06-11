import ipdb
import numpy as np
import theano.tensor as T


def dropout(x, p, theano_rng):
    if p < 0 or p > 1:
        raise ValueError("p should be in [0, 1].")
    mask = theano_rng.binomial(p=p, size=x.shape, dtype=x.dtype)
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
    X       : ndArrays
    overlap : amount of overlap (usually half of the window size)

    Notes
    -----
    This function assumes X as a matrix form of a sequence
    """
    timesteps, frame_size = np.array(X).shape
    new_x = np.zeros(frame_size + (timesteps - 1) * overlap,
                     dtype=np.float32)
    import scipy
    w = scipy.signal.hann(frame_size)
    w2 = w**2
    w_sum = np.zeros(frame_size + (timesteps - 1) * overlap,
                     dtype=np.float32)
    start = 0
    for i in xrange(timesteps):
        new_x[start:start+frame_size] += X[i] * w
        w_sum[start:start+frame_size] += w2
        start += overlap
    w_sum = np.maximum(w_sum, 0.01)
    new_x /= w_sum
    return new_x


def batch_overlap_sum(X, overlap):
    """
    WRITEME

    Parameters
    ----------
    X       : list of lists or ndArrays
    overlap : amount of overlap (usually half of the window size)

    Notes
    -----
    This function assumes X as 3D
    """
    new_X = []
    timesteps, frame_size = np.array(X[0]).shape
    import scipy
    w = scipy.signal.hann(frame_size)
    w2 = w**2
    w_sum = np.zeros(frame_size + (timesteps - 1) * overlap,
                     dtype=np.float32)
    start = 0
    for i in xrange(timesteps):
        w_sum[start:start+frame_size] += w2
        start += overlap
    w_sum = np.maximum(w_sum, 0.01)
    for i in xrange(len(X)):
        timesteps, frame_size = np.array(X[i]).shape
        new_x = np.zeros(frame_size + (timesteps - 1) * overlap,
                         dtype=np.float32)
        start = 0
        for j in xrange(timesteps):
            new_x[start:start+frame_size] += X[i][j] * w
            start += overlap
        new_x /= w_sum
        new_X.append(new_x)
    return np.array(new_X)


def complex_to_real(X):
    """
    WRITEME

    Parameters
    ----------
    X : list of complex vectors

    Notes
    -----
    This function assumes X as 2D
    """
    new_X = []
    for i in xrange(len(X)):
        x = X[i]
        new_x = np.concatenate([np.real(x), np.imag(x)])
        new_X.append(new_x)
    return np.array(new_X)


def real_to_complex(X):
    """
    WRITEME

    Parameters
    ----------
    X : list of complex vectors

    Notes
    -----
    This function assumes X as 2D
    """
    n = X[0].shape[-1]
    new_X = []
    for i in xrange(len(X)):
        x = X[i]
        real = x[:n/2]
        imag = x[n/2:]
        new_x = real + imag*1.0j 
        new_X.append(new_x)
    return np.array(new_X)   
