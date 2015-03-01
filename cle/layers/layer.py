import ipdb
import numpy as np
import theano.tensor as T

from cle.cle.layers import StemCell, RandomCell, InitCell
from cle.cle.utils import tolist, totuple, unpack
from theano.compat.python2x import OrderedDict
from theano.tensor.signal.downsample import max_pool_2d
