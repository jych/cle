import numpy as np
import logging
import theano
import time

logger = logging.getLogger(__name__)


class Monitor(object):
    """
    Monitoring class

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, graph, fn_out=None, freq=100):
        self.graph = graph
        self.fn_out = fn_out
        self._epoch_seen = 0
        self._batch_seen = 0
        self.t0 = 0
        self.freq = freq

    def get_params(self):
        return self.graph.get_params()

    def fprop(self, x=None):
        rlist = []
        for node in self.graph.sorted_nodes:
            rlist.append(node.out)
        return rlist

    def __call__(self, trainlog): 
        t1 = time.time() - self.t0
        logger.info("")
        logger.info("Monitoring step")
        logger.info("***************")
        logger.info("\tTime elapsed: %f" % t1)
        logger.info("\tEpochs  seen: %d" % self._epoch_seen)
        logger.info("\tBatches seen: %d" % self._batch_seen)
        srt = max(0, self._batch_seen - self.freq)
        end = max(1, self._batch_seen)
        for i, out in enumerate(self.fn_out):
            logger.info("\t%s: %f" % (out,
                                      np.asarray(trainlog._batches)[srt:end, i].mean()))


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
