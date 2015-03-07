import ipdb
import logging
import numpy as np
import os
import sys
import theano.tensor as T

from cle.cle.graph import TheanoMixin
from cle.cle.utils import secure_pickle_dump


logger = logging.getLogger(__name__)


class Extension(object):
    def exe(self, mainloop):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError()


class GradientClipping(Extension):
    def __init__(self, scaler=5, batch_size=1):
        """
        .. todo::

            WRITEME
        """
        self.name = 'ext_grad'
        self.scaler = scaler
        self.batch_size = batch_size

    def exe(self, mainloop):
        """
        .. todo::

            WRITEME
        """
        grads = mainloop.grads
        for p, g in grads.items():
            g /= self.batch_size
            g_norm = T.sqrt((g**2).sum())
            not_finite = T.or_(T.isnan(g_norm), T.isinf(g_norm))
            scaler = self.scaler / T.maximum(self.scaler, g_norm)
            grads[p] = T.switch(not_finite, 0.1 * p, g * scaler)
        mainloop.grads = grads


class EpochCount(Extension):
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
        if np.mod(mainloop.trainlog._epoch_seen, self.num_epoch) == 0:
            mainloop.endloop = 1


class Monitoring(Extension, TheanoMixin):
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
            self.monitor_fn = self.build_theano_graph(inputs, self.ddout)
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
                    logger.info("\t%s_%s: %f" %
                                (data.name, ch.name, this_mean))
            mainloop.trainlog._ddmonitors.append(this_ch)
        else:
            pass

    def exe(self, mainloop):
        """
        .. todo::

            WRITEME
        """
        log = mainloop.trainlog
        if np.mod(log._batch_seen, self.freq) == 0:
            srt = max(0, log._batch_seen - self.freq)
            end = max(1, log._batch_seen)
            t = np.asarray(log._times)[srt: end].sum()
            logger.info("")
            logger.info("Monitoring step")
            logger.info("***************")
            logger.info("\tElapsed time: %f" % t)
            logger.info("\tEpochs  seen: %d" % log._epoch_seen)
            logger.info("\tBatches seen: %d" % log._batch_seen)
            optch = [out.name for out in mainloop.outputs]
            for i, out in enumerate(optch):
                this_mean = np.asarray(log._batches)[srt: end, i].mean()
                if this_mean is np.nan:
                    raise ValueError("NaN occured in output.")
                logger.info("\t%s: %f" % (out, this_mean))
            self.monitor_data_based_channels(mainloop)


class Picklize(Extension):
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
        if np.mod(mainloop.trainlog._epoch_seen, self.freq) == 0:
            pklpath = mainloop.name + '.pkl'
            path = os.path.join(self.path, pklpath)
            logger.info("\tSaving model to: %s" % path)
            try:
                secure_pickle_dump(mainloop, path)
            except Exception:
                raise


class EarlyStopping(Extension):
    """
    .. todo::

        WRITEME
    """
    def __init__(self, path):
        self.name = 'ext_save'
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.best = sys.float_info.max

    def exe(self, mainloop):
        """
        Pickle the mainloop
        """
        if len(mainloop.trainlog._ddmonitors) > 0:
            if mainloop.trainlog._ddmonitors[-1][0] < self.best:
                self.best = mainloop.trainlog._ddmonitors[-1][0]
                pklpath = mainloop.name + '_best.pkl'
                path = os.path.join(self.path, pklpath)
                logger.info("\tSaving best model to: %s" % path)
                try:
                    secure_pickle_dump(mainloop, path)
                except Exception:
                    raise
