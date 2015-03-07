import ipdb
import theano.tensor as T


def dropout(x, p, theano_rng):
    if p < 0 or p > 1:
        raise ValueError("p should be in [0, 1].")
    mask = theano_rng.binomial(p=p, size=x.shape,
                               dtype=x.dtype)
    return x * mask


def logsumexp(x, axis=None):
    #x_max = T.max(x, axis=axis, keepdims=True)
    #z = T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    #return z.sum(axis=axis)
    x_max = T.max(x, axis=axis).dimshuffle(0, 'x')
    z = T.log(T.sum(T.exp(x - x_max), axis=axis)) + x_max
    if z.ndim != 1:
        raise ValueError("There is something going wrong with\
                          your dimension after logsumexp.")
    return z


def add_noise(x, stddev, theano_rng):
    x += theano_rng.normal(size=x.shape,
                           avg=0.,
                           std=stddev,
                           dtype=x.dtype)
    return x
