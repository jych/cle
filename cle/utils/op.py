import ipdb
import numpy as np
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.compat.python2x import OrderedDict


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


def add_noise(x, stddev, theano_rg):
    x += theano_rng.normal(size=x.shape,
                               avg=0.,
                               std=stddev,
                               dtype=x.dtype)
    return x
