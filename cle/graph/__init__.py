import theano
import theano.tensor as T

from cle.cle.utils import PickleMixin, OrderedDict, tolist


class TheanoMixin(object):
    """
    WRITEME

    Parameters
    ----------
    .. todo::
    """
    def build_theano_graph(self, inputs, outputs, updates=[]):
        return theano.function(inputs=inputs,
                               outputs=outputs,
                               updates=updates,
                               on_unused_input='ignore',
                               allow_input_downcast=True)
