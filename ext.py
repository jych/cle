import ipdb
import logging
import numpy as np
import os
import theano.tensor as T

from itertools import izip
from util import *


logger = logging.getLogger(__name__)


class GradientClipping(object):
    def __init__(self):
        """
        .. todo::

            WRITEME
        """
        self.name = 'ext_grad'

    def exe(self, mainloop):
        """
        .. todo::

            WRITEME
        """
        grads = mainloop.grads
        g_norm = 0.
        for g in grads.values():
            g /= 128
            g_norm += (g**2).sum()
        not_finite = T.or_(T.isnan(g_norm), T.isinf(g_norm))
        g_norm = T.sqrt(g_norm)
        scaling_num = 5
        scaling_den = T.maximum(5, g_norm)
        r = scaling_num / scaling_den
        for p, g in grads.items():
            grads[p] = T.switch(not_finite, 0.1 * p, g * r)
        mainloop.grads = grads


class EpochCount(object):
    def __init__(self, num_epoch):
        """
        .. todo::

            WRITEME
        """
        self.name = 'ext_term'
        self.num_epoch = num_epoch

    def exe(self, mainloop):
        """
        .. todo::

            WRITEME
        """
        if np.mod(mainloop.trainlog._epoch_seen, self.num_epoch)==0:
            mainloop.endloop = 1


class Monitoring(object):
    def __init__(self, freq, ddout=None, data=None):
        """
        .. todo::

            WRITEME
        """
        self.name = 'ext_monitor'
        self.freq = freq
        self.ddout = ddout
        self.data = data
        self.monitor_fn = None

    def monitor_data_based_channels(self, mainloop):
        """
        .. todo::

            WRITEME
        """
        if self.monitor_fn is None:
            inputs = mainloop.model.get_inputs()
            self.build_computational_graph(inputs, self.ddout)
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
                for i, ch in enumerate(self.ddout):
                    this_mean = record[:, i].mean()
                    if this_mean is np.nan:
                        raise ValueError("NaN occured in output.")
                    this_ch.append(this_mean)
                    logger.info("\t%s_%s: %f" % (data.name, ch.name, this_mean))
            mainloop.trainlog._ddmonitors.append(this_ch)
        else:
            pass

    def build_computational_graph(self, inputs, outputs):
        """
        .. todo::

            WRITEME
        """
        self.monitor_fn = theano.function(inputs, outputs,
                                          on_unused_input='ignore',
                                          allow_input_downcast=True)

    def exe(self, mainloop):
        """
        .. todo::

            WRITEME
        """
        log = mainloop.trainlog
        if np.mod(log._batch_seen, self.freq)==0:
            srt = max(0, log._batch_seen - self.freq)
            end = max(1, log._batch_seen)
            t = np.asarray(log._times)[srt: end].sum()
            logger.info("")
            logger.info("Monitoring step")
            logger.info("***************")
            logger.info("\tTime elapsed: %f" % t)
            logger.info("\tEpochs  seen: %d" % log._epoch_seen)
            logger.info("\tBatches seen: %d" % log._batch_seen)
            optch = [out.name for out in mainloop.outputs]
            for i, out in enumerate(optch):
                this_mean = np.asarray(log._batches)[srt: end, i].mean()
                logger.info("\t%s: %f" % (out, this_mean))
            self.monitor_data_based_channels(mainloop)


class Picklize(object):
    """
    .. todo::

        WRITEME
    """
    def __init__(self, freq, path):
        self.name = 'ext_save'
        self.freq = freq
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

    def exe(self, mainloop):
        """
        Pickle the mainloop
        """
        if np.mod(mainloop.trainlog._epoch_seen, self.freq)==0:
            pklpath = mainloop.name + '.pkl'
            path = os.path.join(self.path, pklpath)
            logger.info("")
            logger.info("Saving model to %s" % path)
            try:
                secure_pickle_dump(mainloop, path)
            except Exception:
                raise            
