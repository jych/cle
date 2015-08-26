import ipdb
import cPickle
import logging
import numpy as np
import os
import shutil
import sys
import tempfile
import theano
import theano.tensor as T

from collections import deque
from numpy.lib.stride_tricks import as_strided
from theano.compat.python2x import OrderedDict


logger = logging.getLogger(__name__)


def topological_sort(graph):
    """
    Topological sort

    Parameters
    ----------
    graph : dict
    """
    GRAY, BLACK = 0, 1
    order, enter, state = deque(), set(graph), {}
    this_graph = dict()
    for node in graph:
        this_graph[node] = tolist(graph[node])

    def dfs(node):
        state[node] = GRAY
        for k in this_graph.get(node, ()):
            sk = state.get(k, None)
            if sk == GRAY:
                raise ValueError("cycle")
            if sk == BLACK:
                continue
            enter.discard(k)
            dfs(k)
        order.appendleft(node)
        state[node] = BLACK
    while enter:
        dfs(enter.pop())
    return order


def one_hot(labels, nlabels=None):
    nlabels = np.max(labels) + 1 if nlabels is None else nlabels
    code = np.zeros((len(labels), nlabels), dtype=theano.config.floatX)
    for i, j in enumerate(labels):
        code[i, j] = 1.
    return code


def T_one_hot(labels, nlabels=None):
    if nlabels is None:
        nlabels = T.max(labels) + 1
    ranges = T.shape_padleft(T.arange(nlabels), labels.ndim)
    return T.cast(T.eq(ranges, T.shape_padright(labels, 1)), 'floatX')


def flatten(nested_list):
    flattened = lambda lst: reduce(lambda l, i: l + flatten(i)
                                   if isinstance(i, (list, tuple))
                                   else l + [i], lst, [])
    return flattened(nested_list)


def uniqify(seq):
    seen = {}
    result = []
    for elem in seq:
        if elem in seen:
            continue
        seen[elem] = 1
        result.append(elem)
    return result


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


def predict(probs, axis=-1):
    return T.argmax(probs, axis=axis)


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
        if isinstance(arg, tuple):
            return list(arg)
        else:
            return [arg]
    return arg


def totuple(arg):
    if type(arg) is not tuple:
        if isinstance(arg, list):
            return tuple(arg)
        else:
            return (arg,)
    return arg


def todict(args):
    rval = OrderedDict()
    if isinstance(args, (list, tuple)):
        for arg in args:
            rval[arg.name] = arg
    else:
        rval[args.name] = args
    return rval


class PickleMixin(object):
    """
    This code is brought from Kyle Kastner.
    """
    def __getstate__(self):
        if not hasattr(self, '_pickle_skip_list'):
            self._pickle_skip_list = []
            self._pickle_skip_list.append('cost_fn')
            self._pickle_skip_list.append('data')
            self._pickle_skip_list.append('extension')
            self._pickle_skip_list.append('grads')
            self._pickle_skip_list.append('params')
            self._pickle_skip_list.append('inputs')
            self._pickle_skip_list.append('outputs')
            self._pickle_skip_list.append('updates')
            self._pickle_skip_list.append('optimizer')
            self._pickle_skip_list.append('endloop')
            self._pickle_skip_list.append('debug_print')
        state = OrderedDict()
        for k, v in self.__dict__.items():
            if k not in self._pickle_skip_list:
                state[k] = v
        return state

    def __setstate__(self, state):
        self.__dict__ = state


def secure_pickle_dump(object_, path):
    """
    This code is brought from Blocks.
    Robust serialization - does not corrupt your files when failed.

    Parameters
    ----------
    object_ : object
        The object to be saved to the disk.
    path : str
        The destination path.
    """
    try:
        d = os.path.dirname(path)
        with tempfile.NamedTemporaryFile(delete=False, dir=d) as temp:
            try:
                cPickle.dump(object_, temp, protocol=-1)
            except RuntimeError as e:
                if str(e).find('recursion') != -1:
                    logger.warning('cle.utils.secure_pickle_dump encountered '
                        'the following error: ' + str(e) +
                        '\nAttempting to resolve this error by calling ' +
                        'sys.setrecusionlimit and retrying')
                    old_limit = sys.getrecursionlimit()
                try:
                    sys.setrecursionlimit(50000)
                    cPickle.dump(object_, temp, protocol=-1)
                finally:
                    sys.setrecursionlimit(old_limit)
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


# push parameters to Theano shared variables
def zipp(params, tparams):

    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):

    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()

    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):

    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(castX(params[kk]), name=kk)

    return tparams


# load parameters
def load_params(path, params):

    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params


class DefaultListOrderedDict(OrderedDict):
    def __missing__(self,k):
        self[k] = []
        return self[k]


def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis
    into overlapping frames.
    This code has been implemented by Anne Archibald and has been discussed
    on the ML.

    Parameters
    ----------
    a : array-like
        The array to segment
    length : int
        The length of each frame
    overlap : int, optional
        The number of array elements by which the frames should overlap
    axis : int, optional
        The axis to operate on; if None, act on the flattened array
    end : {'cut', 'wrap', 'end'}, optional
        What to do with the last frame, if the array is not evenly
        divisible into pieces.

            - 'cut'   Simply discard the extra values
            - 'wrap'  Copy values from the beginning of the array
            - 'pad'   Pad with a constant value

    endvalue : object
        The value to use for end='pad'


    Examples
    --------
    >>> segment_axis(arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

    Notes
    -----
    The array is not copied unless necessary (either because it is
    unevenly strided and being flattened or because end is set to
    'pad' or 'wrap').

    use as_strided

    """

    if axis is None:
        a = np.ravel(a) # may copy
        axis = 0

    l = a.shape[axis]

    if overlap>=length:
        raise ValueError, "frames cannot overlap by more than 100%"
    if overlap<0 or length<=0:
        raise ValueError, "overlap must be nonnegative and length must be "\
                          "positive"

    if l<length or (l-length)%(length-overlap):
        if l>length:
            roundup = length + \
                      (1+(l-length)//(length-overlap))*(length-overlap)
            rounddown = length + \
                        ((l-length)//(length-overlap))*(length-overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown<l<roundup
        assert roundup==rounddown+(length-overlap) or \
               (roundup==length and rounddown==0)
        a = a.swapaxes(-1,axis)

        if end=='cut':
            a = a[...,:rounddown]
        elif end in ['pad','wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1]=roundup
            b = np.empty(s,dtype=a.dtype)
            b[...,:l] = a
            if end=='pad':
                b[...,l:] = endvalue
            elif end=='wrap':
                b[...,l:] = a[...,:roundup-l]
            a = b

        a = a.swapaxes(-1,axis)


    l = a.shape[axis]
    if l==0:
        raise ValueError, "Not enough data points to segment array in 'cut' "\
                          "mode; try 'pad' or 'wrap'"
    assert l>=length
    assert (l-length)%(length-overlap) == 0
    n = 1+(l-length)//(length-overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n,length) + a.shape[axis+1:]
    newstrides = a.strides[:axis] + ((length-overlap)*s, s) + \
                 a.strides[axis+1:]

    try:
        return as_strided(a, strides=newstrides, shape=newshape)
    except TypeError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length-overlap)*s, s) + \
                     a.strides[axis+1:]
        return as_strided(a, strides=newstrides, shape=newshape)
