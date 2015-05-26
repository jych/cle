import ipdb
import cPickle
import logging
import numpy as np
import os
import sys
import theano
import theano.tensor as T
import time

from cle.cle.graph import TheanoMixin
from cle.cle.utils import secure_pickle_dump, tolist


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
        """
        for p, g in grads.items():
            grads[p] = g / self.batch_size
        g_norm = 0.
        for g in grads.values():
            g_norm += (g**2).sum()
        """
        g_norm = 0.
        for p, g in grads.items():
            g /= T.cast(self.batch_size, dtype=theano.config.floatX)
            grads[p] = g
            g_norm += (g**2).sum()
        not_finite = T.or_(T.isnan(g_norm), T.isinf(g_norm))
        g_norm = T.sqrt(g_norm)
        scaler = self.scaler / T.maximum(self.scaler, g_norm)
        for p, g in grads.items():
            grads[p] = T.switch(not_finite, 0.1 * p, g * scaler)
        mainloop.grads = grads

#    def exe(self, mainloop):
#        """
#        .. todo::
#
#            WRITEME
#        """
#        grads = mainloop.grads
#        g_norm = 0.
#        for g in grads.values():
#            g /= self.batch_size
#            g_norm += (g**2).sum()
#        not_finite = T.or_(T.isnan(g_norm), T.isinf(g_norm))
#        g_norm = T.sqrt(g_norm)
#        scaler = self.scaler / T.maximum(self.scaler, g_norm)
#        for p, g in grads.items():
#            g /= self.batch_size
#            grads[p] = T.switch(not_finite, 0.1 * p, g * scaler)
#        mainloop.grads = grads


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
    def __init__(self, freq, ddout=None, data=None, monitor_fn=None):
        """
        .. todo::

            WRITEME
        """
        self.name = 'ext_monitor'
        self.freq = freq
        self.ddout = ddout
        self.data = data
        self.monitor_fn = monitor_fn

    def monitor_data_based_channels(self, mainloop):
        """
        .. todo::

            WRITEME
        """
        if self.monitor_fn is None:
            inputs = mainloop.inputs
            self.monitor_fn = self.build_theano_graph(inputs, self.ddout)
        if self.data is not None:
            data_record = []
            for data in self.data:
                batch_record = []
                for batch in data:
                    this_out = self.monitor_fn(*batch)
                    batch_record.append(this_out)
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
            logger.info("\t----------------")
            logger.info("\tTraininig basics")
            logger.info("\t................")
            logger.info("\tElapsed time: %f" % t)
            logger.info("\tEpochs  seen: %d" % log._epoch_seen)
            logger.info("\tBatches seen: %d" % log._batch_seen)
            logger.info("\t-----------------------")
            logger.info("\tOptimization parameters")
            logger.info("\t.......................")
            mainloop.optimizer.monitor()
            logger.info("\t------------------")
            logger.info("\tForward-prop based")
            logger.info("\t..................")
            output_channel = [out.name for out in mainloop.outputs]
            if log._batch_seen == 0:
                logger.info("\tinitial_monitoring")
            else:
                for i, out in enumerate(output_channel):
                    this_mean = np.asarray(log._batches)[srt: end, i].mean()
                    if this_mean is np.nan:
                        raise ValueError("NaN occured in output.")
                    logger.info("\tthis_batch_%s: %f" % (out, this_mean))
            this_t0 = time.time()
            self.monitor_data_based_channels(mainloop)
            mt = time.time() - this_t0
            logger.info("\tElapsed time for monitoring: %f" % mt)


class Picklize(Extension):
    """
    .. todo::

        WRITEME
    """
    def __init__(self, freq, path, force_save_freq=1e15):
        self.name = 'ext_save'
        self.freq = freq
        self.force_save_freq = force_save_freq
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

    def exe(self, mainloop):
        """
        Pickle the mainloop
        """
        if np.mod(mainloop.trainlog._batch_seen, self.freq) == 0:
            pkl_path = mainloop.name + '.pkl'
            path = os.path.join(self.path, pkl_path)
            logger.info("\tSaving model to: %s" % path)
            try:
                import sys
                sys.setrecursionlimit(50000)
                f = open(path, 'wb')
                cPickle.dump(mainloop, f, -1)
                f.close()
                #secure_pickle_dump(mainloop, path)
            except Exception:
                raise
        if np.mod(mainloop.trainlog._batch_seen, self.force_save_freq) == 0:
            force_pkl_path = mainloop.name + '_' +\
                             str(mainloop.trainlog._batch_seen) +\
                             'updates.pkl'
            force_path = os.path.join(self.path, force_pkl_path)
            logger.info("\tSaving model to: %s" % force_path)
            try:
                import sys
                sys.setrecursionlimit(50000)
                f = open(force_path, 'wb')
                cPickle.dump(mainloop, f, -1)
                f.close()
                #secure_pickle_dump(mainloop, path)
            except Exception:
                raise


class EarlyStopping(Extension):
    """
    .. todo::

        WRITEME
    """
    def __init__(self, path, freq=1, force_save_freq=None):
        self.name = 'ext_save'
        self.freq = freq
        self.force_save_freq = force_save_freq
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
                if np.mod(mainloop.trainlog._batch_seen, self.freq) == 0:
                    self.best = mainloop.trainlog._ddmonitors[-1][0]
                    pkl_path = mainloop.name + '_best.pkl'
                    path = os.path.join(self.path, pkl_path)
                    logger.info("\tSaving best model to: %s" % path)
                    try:
                        import sys
                        sys.setrecursionlimit(50000)
                        f = open(path, 'wb')
                        cPickle.dump(mainloop, f, -1)
                        f.close()
                        #secure_pickle_dump(mainloop, path)
                    except Exception:
                        raise
                    if self.force_save_freq is not None:
                        this_scaler = (mainloop.trainlog._batch_seen /
                                      self.force_save_freq)
                        this_number = self.force_save_freq * (this_scaler + 1)
                        force_pkl_path = mainloop.name + '_best_before_' +\
                                         str(this_number) +\
                                         'updates.pkl'
                        force_path = os.path.join(self.path, force_pkl_path)
                        logger.info("\tSaving best model to: %s" % force_path)
                        try:
                            import sys
                            sys.setrecursionlimit(50000)
                            f = open(force_path, 'wb')
                            cPickle.dump(mainloop, f, -1)
                            f.close()
                            #secure_pickle_dump(mainloop, path)
                        except Exception:
                            raise


class WeightDecay(Extension):
    def __init__(self, lambd=0.0002, param_name=['W']):
        """
        .. todo::

            WRITEME
        """
        self.name = 'ext_regularize_pre_grad'
        self.lambd = lambd
        self.param_name = tolist(param_name)

    def exe(self, mainloop):
        """
        .. todo::

            WRITEME
        """
        for p in mainloop.params:
            for pname in self.param_name:
                if pname in p.name:
                    mainloop.cost += self.lambd * 0.5 * (p**2).sum()


class WeightNorm(Extension):
    def __init__(self, is_vector=1, weight_norm=1.9365, param_name=['W']):
        """
        .. todo::

            WRITEME
        """
        self.name = 'ext_regularize_post_grad'
        self.weight_norm = weight_norm
        self.param_name = tolist(param_name)
        self.is_vector = is_vector

    def exe(self, mainloop):
        """
        .. todo::

            WRITEME
        """
        for k, p in mainloop.updates.items():
            for pname in self.param_name:
                if pname in str(k):
                    updated_W = mainloop.updates[k]
                    if self.is_vector:
                        col_norms = T.sqrt(T.sqr(updated_W).sum(axis=0))
                        desired_norms = T.clip(col_norms, 0, self.weight_norm)
                        ratio = (desired_norms / (1e-7 + col_norms))
                        mainloop.updates[k] = updated_W * ratio
                    else:
                        norm = T.sqrt(T.sqr(updated_W).sum())
                        desired_norm = T.clip(norm, 0, self.weight_norm)
                        ratio = (desired_norm / (1e-7 + norm))
                        mainloop.updates[k] = updated_W * ratio
