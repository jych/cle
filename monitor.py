import ipdb
import numpy as np
import logging
import theano
import time

logger = logging.getLogger(__name__)


class Monitor(object):
    """
    Monitoring class

    There are optimization based channels and data-driven channels.
    ... [opt_ch, dd_ch]
    ... The former is mainly about training time and number of iterations.
    ... The latter is cost or log-likelihood of each dataset channel.
    Monitor doesn't support observation of activations or hidden states.

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 model,
                 freq=100,
                 #opt_ch=None,
                 dd_ch=None,
                 data=None,
                 channels=None):
        self.model = model
        self.freq = freq
        #self.opt_ch = opt_ch
        self.dd_ch = dd_ch
        if dd_ch is not None:
            if data is None:
                raise AssertionError("You should provide data.")
        self.data = data
        self.channels = channels

        self._epoch_seen = 0
        self._batch_seen = 0
        self.t0 = 0

        self._dd_ch = []

    def monitor_data_based_channels(self):
        if self.data is not None:
            data_record = []
            for data in self.data:
                batch_record = []
                for batch in data:
                    this_cost = self.monitor_fn(*batch)
                    batch_record.append(this_cost)
                data_record.append(np.asarray(batch_record)) 
            this_ch = []
            for record, data in zip(data_record, self.data):
                for i, ch in enumerate(self.dd_ch):
                    this_mean = record[:, i].mean()
                    this_ch.append(this_mean)
                    logger.info("\t%s_%s: %f" % (data.name, ch, this_mean))
            self._dd_ch.append(this_ch)
        else:
            pass

    def build_monitor_graph(self, outputs):
        if len(outputs) != len(self.dd_ch):
            raise AssertionError("Number of outputs and channels should match")
        inputs = self.model.get_inputs()
        self.monitor_fn = self.get_theano_graph(inputs, outputs)

    def get_theano_graph(self, inputs, outputs):
        return theano.function(inputs=inputs,
                               outputs=outputs,
                               on_unused_input='ignore',
                               allow_input_downcast=True)

    def get_params(self):
        return self.model.get_params()

    def __call__(self, trainlog): 
        t1 = time.time() - self.t0
        logger.info("")
        logger.info("Monitoring step")
        logger.info("***************")
        logger.info("\tTime elapsed: %f" % t1)
        logger.info("\tEpochs  seen: %d" % self._epoch_seen)
        logger.info("\tBatches seen: %d" % self._batch_seen)
        #srt = max(0, self._batch_seen - self.freq)
        #end = max(1, self._batch_seen)
        #for i, out in enumerate(self.opt_ch):
        #    this_mean = np.asarray(trainlog._batches)[srt: end, i].mean()
        #    logger.info("\t%s: %f" % (out, this_mean))
        self.monitor_data_based_channels()


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
