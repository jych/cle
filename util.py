import cPickle
import numpy as np
import os
import shutil
import tempfile
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


def predict(probs):
    return T.argmax(probs, axis=-1)


def error(labels, pred_labels):
    return T.mean(T.neq(pred_labels, labels))


def unpack(arg):
    if isinstance(arg, (list, tuple)):
        if len(arg) == 1:
            return arg[0]
        else:
            return list[arg]
    elif type(arg) is np.ndarray:
        return arg[0]
    else:
        return arg

def tolist(arg):
    if type(arg) is not list:
        arg = [arg]
    return arg


class PickleMixin(object):
    """
    This code is brought from Kyle Kastner

    ----------
    .. todo::
    """
    def __getstate__(self):
        if not hasattr(self, '_pickle_skip_list'):
            self._pickle_skip_list = []
            for k, v in self.__dict__.items():
                try:
                    f = tempfile.TemporaryFile()
                    cPickle.dump(v, f)
                except:
                    self._pickle_skip_list.append(k)
        state = OrderedDict()
        for k, v in self.__dict__.items():
            if k not in self._pickle_skip_list:
                state[k] = v
        return state
 
    def __setstate__(self, state):
        self.__dict__ = state


def secure_pickle_dump(object_, path):
    """
    This code is brought from Blocks
    Robust serialization - does not corrupt your files when failed.

    Parameters
    ----------
    object_ : object
        The object to be saved to the disk.
    path : str
        The destination path.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            cPickle.dump(object_, temp)
        shutil.move(temp.name, path)
    except:
        if "temp" in locals():
            os.remove(temp.name)
        raise


def unpickle(path):
    """
    .. todo::

        WRITEME
    """
    f = open(path, 'rb')
    m = cPickle.load(f)
    f.close()
    return m


def initialize_from_pkl(arg, path):
    """
    .. todo::

        WRITEME
    """
    f = open(path, 'rb')
    m = cPickle.load(f)
    arg.__setstate__(m.__dict__)
    f.close()
