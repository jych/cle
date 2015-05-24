import ipdb
import logging
import theano.tensor as T
import time

from itertools import izip
from cle.cle.graph import TheanoMixin
from cle.cle.models import Model
from cle.cle.utils import PickleMixin, OrderedDict, tolist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Training(PickleMixin, TheanoMixin):
    """
    WRITEME

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 name,
                 data,
                 model,
                 optimizer,
                 cost,
                 outputs,
                 debug_print=0,
                 extension=None):
        self.name = name
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.inputs = model.inputs
        self.params = model.params
        self.cost = cost
        self.outputs = tolist(outputs)
        self.updates = model.updates
        self.extension = extension
        self.debug_print = debug_print

        t0 = time.time()
        self.cost_fn = self.build_training_graph()
        print "Elapsed compilation time: %f" % (time.time() - t0)
        if self.debug_print:
            from theano.printing import debugprint
            debugprint(self.cost_fn)
        self.trainlog = TrainLog()
        self.endloop = 0

    def build_training_graph(self):
        self.run_extension('ext_regularize_pre_grad')
        self.grads = OrderedDict(izip(self.params,
                                      T.grad(self.cost, self.params)))
        self.run_extension('ext_grad')
        grads = self.optimizer.get_updates(self.grads)
        for key, val in grads.items():
            self.updates[key] = val
        self.run_extension('ext_regularize_post_grad')
        return self.build_theano_graph(self.inputs, self.outputs, self.updates)

    def run(self):
        logger.info("Entering main loop")
        while self.run_epoch():
            pass
        logger.info("Terminating main loop")

    def run_epoch(self):
        for batch in self.data:
            self.run_extension('ext_monitor')
            batch_t0 = time.time()
            this_cost = self.cost_fn(*batch)
            self.trainlog._times.append(time.time() - batch_t0)
            self.trainlog._batches.append(this_cost)
            self.trainlog._batch_seen += 1
            #self.run_extension('ext_monitor')
            self.run_extension('ext_save')
        self.trainlog._epoch_seen += 1
        self.run_extension('ext_term')
        if self.end_training():
            return False
        return True

    def find_extension(self, name):
        try:
            exts = [extension for extension in self.extension
                    if extension.name == name]
            if len(exts) > 0:
                return_val = 1
            else:
                return_val = 0
            return return_val, exts
        except:
            return (0, None)

    def run_extension(self, name):
        tok, exts = self.find_extension(name)
        if tok:
            for ext in exts:
                ext.exe(self)

    def end_training(self):
        return self.endloop


class TrainLog(object):
    """
    Training log class

    Parameters
    ----------
    .. todo::
    """
    def __init__(self):
        self._batches = []
        self._times = []
        self._ddmonitors = []
        self._epoch_seen = 0
        self._batch_seen = 0
