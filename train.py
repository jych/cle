import ipdb
import logging
import theano
import time

from itertools import izip
from layer import *
from monitor import Monitor, TrainLog
from opt import *
from util import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Training(object):
    def __init__(self,
                 data,
                 model,
                 optimizer,
                 inputs,
                 outputs=None,
                 monitor=None,
                 extension=None):
        self.data = data
        self.model = model
        self.optimizer = optimizer
        if monitor is None:
            self.monitor = Monitor(model)
        self.monitor = monitor
        self.extension = extension

        self.inputs = inputs
        self.outputs = outputs
        self.cost_fn = self.build_training_graph()

        self.trainlog = TrainLog()

    def get_theano_graph(self,
                         outputs=None,
                         updates=[]):
        if outputs is None:
            outputs = self.outputs
        if outputs is not list:
            outputs = [outputs]
        return theano.function(inputs=self.inputs,
                               outputs=outputs,
                               updates=updates,
                               on_unused_input='ignore',
                               allow_input_downcast=True)

    def find_extension(self, name):
        try:
            exts = [extension for extension in self.extension
                    if extension.name==name]
            if len(exts) > 0:
                return_val = 1
            else:
                return_val = 0
            return return_val, exts
        except:
            return (0, None)
        
    def build_training_graph(self):
        cost = self.model.nodes['cost'].out
        grads = OrderedDict(izip(self.model.params,
                                 T.grad(cost, self.model.params)))
        tok, exts = self.find_extension('ext_grads')
        if tok:
            for ext in exts:
                grads = ext.apply(grads)
        updates = self.optimizer.get_updates(grads)
        return self.get_theano_graph(cost, updates)

    def run(self):
        logger.info("Starting main loop")    
        while self.run_epoch():
            pass
                 
    def run_epoch(self):
        while self.run_batch():
            pass
        self.monitor._epoch_seen += 1
        self.check_termination_criteria()
        return True

    def run_batch(self):
        try:
            batch = self.data.next()
        except:
            return False
        batch_t0 = time.time()
        this_cost = self.cost_fn(*batch)
        batch_t1 = time.time() - batch_t0
        self.trainlog._times.append(batch_t1)
        self.trainlog._batches.append(this_cost)
        self.monitor._batch_seen += 1
        if np.mod(self.monitor._batch_seen, self.monitor.freq)==0 or\
            self.monitor._batch_seen==1:
            self.monitor.t0 = time.time()
            self.monitor(self.trainlog)
        return True

    def check_termination_criteria(self):
        tok, exts = self.find_extension('ext_terms')
        if tok:
            for ext in exts:
                if ext.validate():
                    raise TrainingEnd

class TrainingEnd(Exception):
    pass
