import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.compat.python2x import OrderedDict


rng = np.random.RandomState((2015, 2, 19))


def one_hot(labels, nC=None):
    nC = np.max(labels) + 1 if nC is None else nC
    code = np.zeros((len(labels), nC), dtype='float32')
    for i, j in enumerate(labels):
        code[i, j] = 1.
    return code


def flatten(nested_list):
    flattened_list = []
    for x in nested_list:
        flattened_list.extend(x)
    return flattened_list


def castX(value):
    return theano._asarray(value, dtype=theano.config.floatX)


def sharedX(value, name=None, borrow=False):
    return theano.shared(castX(value), name=name, borrow=borrow)


def shared_int(value, dtype='int32', name=None, borrow=False):
    theano_args = theano._asarray(value,
                                  dtype=dtype,
                                  name=name,
                                  borrow=borrow)
    return theano.shared(theano_args)


def dropout(x, p, rng):
    theano_rng = MRG_RandomStreams(max(rng.randint(2 ** 15), 1))
    assert 0 <= p and p < 1
    return T.switch(
        theano_rng.binomial(p=1-p, size=x.shape, dtype=x.dtype),
        x/(1-p), 0.*x
    )


def NLL_mul(probs, targets):
    return - T.sum(targets * T.log(probs)) / probs.shape[0]


def NLL_bin(probs, targets):
    return - T.sum(targets * T.log(probs) + (1-targets) * T.log(1-probs)) / probs.shape[0]


def predict(probs):
    return T.argmax(probs, axis=1)


def error(labels, pred_labels):
    return T.mean(T.neq(pred_labels, labels))
