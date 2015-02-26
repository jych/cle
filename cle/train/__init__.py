import ipdb
import logging
import theano
import theano.tensor as T
import time

from itertools import izip
from cle.cle.graph import TheanoMixin
from cle.cle.util import PickleMixin, OrderedDict, tolist

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
                 extension=None):
        self.name = name
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.inputs = model.get_inputs()
        self.cost = cost
        self.outputs = tolist(outputs)
        self.extension = extension

        self.cost_fn = self.build_training_graph()
        self.trainlog = TrainLog()

        self.endloop = 0

    def build_training_graph(self):
        self.grads = OrderedDict(izip(self.model.params,
                                      T.grad(self.cost, self.model.params)))
        self.run_extension('ext_grad')
        updates = self.optimizer.get_updates(self.grads)
        return self.build_theano_graph(self.inputs, self.outputs, updates)

    def run(self):
        logger.info("Entering main loop")
        while self.run_epoch():
            pass

    def run_epoch(self):
        while self.run_batch():
            pass
        self.trainlog._epoch_seen += 1
        self.run_extension('ext_term')
        self.run_extension('ext_save')
        if self.end_training():
            return False
        return True

    def run_batch(self):
        try:
            batch = self.data.next()
        except:
            return False
        batch_t0 = time.time()
        this_cost = self.cost_fn(*batch)
        self.trainlog._times.append(time.time() - batch_t0)
        self.trainlog._batches.append(this_cost)
        self.trainlog._batch_seen += 1
        self.run_extension('ext_monitor')
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


class TrainingEnd(Exception):
    pass


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
