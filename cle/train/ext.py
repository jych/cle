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
    def __init__(self, scaler=5, batch_size=1, check_nan=0):
        """
        .. todo::

            WRITEME
        """
        self.name = 'ext_grad'
        self.scaler = scaler
        self.batch_size = batch_size
        self.check_nan = check_nan

    def exe(self, mainloop):
        """
        .. todo::

            WRITEME
        """
        grads = mainloop.grads
        g_norm = 0.
        for p, g in grads.items():
            g /= T.cast(self.batch_size, dtype=theano.config.floatX)
            grads[p] = g
            g_norm += (g**2).sum()
        if self.check_nan:
            not_finite = T.or_(T.isnan(g_norm), T.isinf(g_norm))
        g_norm = T.sqrt(g_norm)
        scaler = self.scaler / T.maximum(self.scaler, g_norm)
        if self.check_nan:
            for p, g in grads.items():
                grads[p] = T.switch(not_finite, 0.1 * p, g * scaler)
        else:
            for p, g in grads.items():
                grads[p] = g * scaler
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
        if np.mod(mainloop.trainlog.epoch_seen, self.num_epoch) == 0:
            mainloop.endloop = 1


class Monitoring(Extension, TheanoMixin):
    def __init__(self, freq, ddout=None, data=None, monitor_fn=None,
            obj_monitor_fn=None, obj_monitor_ch=[]):
        """
        obj_monitor_fn :
            Python function, a function adapted to the mean of main objective,
            e.g., perplexity
        .. todo::

            WRITEME
        """
        self.name = 'ext_monitor'
        self.freq = freq
        self.ddout = ddout
        self.data = data
        self.monitor_fn = monitor_fn
        self.obj_monitor_fn = obj_monitor_fn
        self.obj_monitor_ch = obj_monitor_ch

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
            for record, data in zip(data_record, self.data):
                for i, ch in enumerate(self.ddout):
                    this_mean = record[:, i].mean()
                    if this_mean is np.nan:
                        raise ValueError("NaN occured in output.")
                    logger.info(" %s_%s: %f" %
                                (data.name, ch.name, this_mean))
                    ch_name = "%s_%s" % (data.name, ch.name)
                    mainloop.trainlog.monitor[ch_name].append(this_mean)
                    if i < len(self.obj_monitor_ch) and self.obj_monitor_fn is not None:
                        obj_monitor_val = self.obj_monitor_fn(this_mean)
                        ch_name = "%s_%s" % (data.name, self.obj_monitor_ch[i])
                        logger.info(" %s: %f" % (ch_name, obj_monitor_val))
                        mainloop.trainlog.monitor[ch_name].append(obj_monitor_val)
        else:
            pass

    def exe(self, mainloop):
        """
        .. todo::

            WRITEME
        """
        log = mainloop.trainlog
        if np.mod(log.batch_seen, self.freq) == 0 or mainloop.endloop:
            srt = max(0, log.batch_seen - self.freq)
            end = max(1, log.batch_seen)
            t = np.asarray(log.monitor['time'])[srt: end].sum()
            logger.info("")
            logger.info(" Monitoring step")
            logger.info(" ***************")
            logger.info(" ----------------")
            logger.info(" Traininig basics")
            logger.info(" ................")
            logger.info(" Elapsed time: %f" % t)
            logger.info(" Epochs  seen: %d" % log.epoch_seen)
            logger.info(" Batches seen: %d" % log.batch_seen)
            logger.info(" -----------------------")
            logger.info(" Optimization parameters")
            logger.info(" .......................")
            mainloop.optimizer.monitor()
            logger.info(" ------------------")
            logger.info(" Forward-prop based")
            logger.info(" ..................")
            output_channel = [out.name for out in mainloop.outputs]
            if log.batch_seen == 0:
                logger.info(" initial_monitoring")
            else:
                for i, out in enumerate(output_channel):
                    this_mean = np.asarray(log.monitor['update'])[srt: end, i].mean()
                    if this_mean is np.nan:
                        raise ValueError("NaN occured in output.")
                    logger.info(" this_batch_%s: %f" % (out, this_mean))
            this_t0 = time.time()
            self.monitor_data_based_channels(mainloop)
            mt = time.time() - this_t0
            logger.info(" Elapsed time for monitoring: %f" % mt)


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
        if np.mod(mainloop.trainlog.batch_seen, self.freq) == 0 or mainloop.endloop:
            pkl_path = mainloop.name + '.pkl'
            path = os.path.join(self.path, pkl_path)
            logger.info(" Saving model to: %s" % path)
            try:
                import sys
                sys.setrecursionlimit(50000)
                f = open(path, 'wb')
                cPickle.dump(mainloop, f, -1)
                f.close()
                #secure_pickle_dump(mainloop, path)
            except Exception:
                raise
        if np.mod(mainloop.trainlog.batch_seen, self.force_save_freq) == 0:
            force_pkl_path = mainloop.name + '_' +\
                             str(mainloop.trainlog.batch_seen) +\
                             'updates.pkl'
            force_path = os.path.join(self.path, force_pkl_path)
            logger.info(" Saving model to: %s" % force_path)
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
    def __init__(self, path, channel=None, freq=1, force_save_freq=None,
                 minimize=1):
        self.name = 'ext_save'
        self.freq = freq
        self.force_save_freq = force_save_freq
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.best = sys.float_info.max
        self.minimize_ = minimize
        if not self.minimize_:
            self.best *= -1
        self.channel = channel
        if self.channel is None:
            self.channel = 'valid_nll'
            #raise AttributeError("channel is required for early stopping.")

    def exe(self, mainloop):
        """
        Pickle the mainloop
        """
        if len(mainloop.trainlog.monitor['update']) > 0:
            if np.mod(mainloop.trainlog.batch_seen, self.freq) == 0 or mainloop.endloop:
                token = 0
                if self.minimize_:
                    if mainloop.trainlog.monitor[self.channel][-1] < self.best:
                        token = 1
                else:
                    if mainloop.trainlog.monitor[self.channel][-1] > self.best:
                        token = 1
                if token:
                    self.best = mainloop.trainlog.monitor[self.channel][-1]
                    pkl_path = mainloop.name + '_best.pkl'
                    path = os.path.join(self.path, pkl_path)
                    logger.info(" Saving best model to: %s" % path)
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
                        this_scaler = (mainloop.trainlog.batch_seen /
                                      self.force_save_freq)
                        this_number = self.force_save_freq * (this_scaler + 1)
                        force_pkl_path = mainloop.name + '_best_before_' +\
                                         str(this_number) +\
                                         'updates.pkl'
                        force_path = os.path.join(self.path, force_pkl_path)
                        logger.info(" Saving best model to: %s" % force_path)
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


class LrLinearDecay(Extension):
    def __init__(self, start, end, decay_factor):
        """
        .. todo::

            WRITEME
        """
        self.name = 'ext_schedule'
        assert start > 0
        assert end > start
        self.start = start
        self.end = end
        self.decay_factor = decay_factor
        self.count = 0

    def exe(self, mainloop):
        """
        .. todo::

            WRITEME
        """
        if self.count == 0:
            self.base_lr = mainloop.optimizer.lr.get_value()
            self.step = ((self.base_lr - self.base_lr * self.decay_factor) /
                         (self.end - self.start + 1))

        self.count += 1

        if self.count >= self.start:
            if self.count < self.end:
                new_lr = self.base_lr - self.step * (self.count - self.start + 1)
            else:
                new_lr = self.base_lr * self.decay_factor
        else:
            new_lr = self.base_lr

        assert new_lr > 0
        new_lr = np.cast[theano.config.floatX](new_lr)
        mainloop.optimizer.lr.set_value(new_lr)


class LrExponentialDecay(Extension):
    def __init__(self, decay_factor, min_lr):
        """
        .. todo::

            WRITEME
        """
        self.name = 'ext_schedule'
        self.count = 0
        self.decay_factor = decay_factor
        self.min_ = False

    def exe(self, mainloop):
        """
        .. todo::

            WRITEME
        """
        if self.count == 0:
            self.base_lr = mainloop.optimizer.lr.get_value()
        self.count += 1

        if not self.min_:
            new_lr = self.base_lr / (self.decay_factor ** self.count)
            if new_lr <= self.min_lr
                self.min_ = True
                new_lr = self.min_lr
        else:
            new_lr = self.min_lr

        new_lr = np.cast[theano.config.floatX](new_lr)
        mainloop.optimizer.lr.set_value(new_lr)
