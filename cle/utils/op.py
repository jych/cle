import ipdb
import numpy as np
import theano.tensor as T

from cle.cle.utils import predict

from theano.compat.python2x import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams


seed_rng = np.random.RandomState(np.random.randint(1024))
theano_seed = seed_rng.randint(np.iinfo(np.int32).max)
default_theano_rng = MRG_RandomStreams(theano_seed)


def dropout(x, p=0.5, theano_rng=default_theano_rng):
    if p < 0 or p > 1:
        raise ValueError("p should be in [0, 1].")
    mask = theano_rng.binomial(p=p, size=x.shape, dtype=x.dtype)
    return x * mask


def logsumexp(x, axis=None):
    x_max = T.max(x, axis=axis, keepdims=True)
    z = T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return z.sum(axis=axis)


def add_noise(x, std_dev=0.075, theano_rng=default_theano_rng):
    x += theano_rng.normal(size=x.shape, avg=0., std=std_dev, dtype=x.dtype)
    return x


def add_noise_params(params, keys=['W'], std_dev=0.075):

    nparams = OrderedDict()

    for param in params.items():
        key_in = 0
        for key in keys:
            if key in param[0]:
                nparams[param[0]] = add_noise(param[1].copy(), std_dev=std_dev)
                key_in = 1
        if not key_in:
            nparams[param[0]] = param[1].copy()

    return nparams


def Gaussian_sample(mu, sig, num_sample=None, theano_rng=default_theano_rng):

    if num_sample is None:
        num_sample = 1

    mu = mu.dimshuffle(0, 'x', 1)
    sig = sig.dimshuffle(0, 'x', 1)
    epsilon = theano_rng.normal(size=(mu.shape[0],
                                      num_sample,
                                      mu.shape[-1]),
                                avg=0., std=1.,
                                dtype=mu.dtype)
    z = mu + sig * epsilon

    if num_sample == 1:
        z = z.reshape((z.shape[0] * z.shape[1], -1))

    return z


def GMM_sample(mu, sig, coeff, theano_rng=default_theano_rng):

    mu = mu.reshape((mu.shape[0],
                     mu.shape[1]/coeff.shape[-1],
                     coeff.shape[-1]))

    sig = sig.reshape((sig.shape[0],
                       sig.shape[1]/coeff.shape[-1],
                       coeff.shape[-1]))

    idx = predict(
        theano_rng.multinomial(
            pvals=coeff,
            dtype=coeff.dtype
        ),
        axis=1
    )

    mu = mu[T.arange(mu.shape[0]), :, idx]
    sig = sig[T.arange(sig.shape[0]), :, idx]
    epsilon = theano_rng.normal(size=mu.shape,
                                avg=0., std=1.,
                                dtype=mu.dtype)

    z = mu + sig * epsilon

    return z


def GMM_argmax_mean(mu, sig, coeff, theano_rng=default_theano_rng):

    mu = mu.reshape((mu.shape[0],
                     mu.shape[1]/coeff.shape[-1],
                     coeff.shape[-1]))

    sig = sig.reshape((sig.shape[0],
                       sig.shape[1]/coeff.shape[-1],
                       coeff.shape[-1]))

    idx = predict(coeff)
    mu = mu[T.arange(mu.shape[0]), :, idx]
    sig = sig[T.arange(sig.shape[0]), :, idx]

    epsilon = theano_rng.normal(size=mu.shape,
                                avg=0., std=1.,
                                dtype=mu.dtype)

    z = mu + sig * epsilon

    return z, mu


def GMM_sample_mean(mu, sig, coeff, theano_rng=default_theano_rng):

    mu = mu.reshape((mu.shape[0],
                     mu.shape[1]/coeff.shape[-1],
                     coeff.shape[-1]))

    sig = sig.reshape((sig.shape[0],
                       sig.shape[1]/coeff.shape[-1],
                       coeff.shape[-1]))

    idx = predict(
        theano_rng.multinomial(
            pvals=coeff,
            dtype=coeff.dtype
        ),
        axis=1
    )

    mu = mu[T.arange(mu.shape[0]), :, idx]
    sig = sig[T.arange(sig.shape[0]), :, idx]

    epsilon = theano_rng.normal(size=mu.shape,
                                avg=0., std=1.,
                                dtype=mu.dtype)

    z = mu + sig * epsilon

    return z, mu


def overlap_sum(X, overlap):
    """
    WRITEME

    Parameters
    ----------
    X       : ndArrays
    overlap : amount of overlap (usually half of the window size)

    Notes
    -----
    This function assumes X as a matrix form of a sequence
    """
    timesteps, frame_size = np.array(X).shape
    new_x = np.zeros(frame_size + (timesteps - 1) * overlap,
                     dtype=np.float32)
    import scipy
    w = scipy.signal.hann(frame_size)
    w2 = w**2
    w_sum = np.zeros(frame_size + (timesteps - 1) * overlap,
                     dtype=np.float32)
    start = 0
    for i in xrange(timesteps):
        new_x[start:start+frame_size] += X[i] * w
        w_sum[start:start+frame_size] += w2
        start += overlap
    w_sum = np.maximum(w_sum, 0.01)
    new_x /= w_sum
    return new_x


def batch_overlap_sum(X, overlap):
    """
    WRITEME

    Parameters
    ----------
    X       : list of lists or ndArrays
    overlap : amount of overlap (usually half of the window size)

    Notes
    -----
    This function assumes X as 3D
    """
    new_X = []
    timesteps, frame_size = np.array(X[0]).shape
    import scipy
    w = np.maximum(scipy.signal.hann(frame_size), 1e-4)
    w2 = w**2
    w_sum = np.zeros(frame_size + (timesteps - 1) * overlap,
                     dtype=np.float32)
    start = 0
    for i in xrange(timesteps):
        w_sum[start:start+frame_size] += w2
        start += overlap
    for i in xrange(len(X)):
        timesteps, frame_size = np.array(X[i]).shape
        new_x = np.zeros(frame_size + (timesteps - 1) * overlap,
                         dtype=np.float32)
        start = 0
        for j in xrange(timesteps):
            new_x[start:start+frame_size] += X[i][j] * w
            start += overlap
        new_x /= w_sum
        new_X.append(new_x)
    return np.array(new_X)


def complex_to_real(X):
    """
    WRITEME

    Parameters
    ----------
    X : list of complex vectors

    Notes
    -----
    This function assumes X as 2D
    """
    new_X = []
    for i in xrange(len(X)):
        x = X[i]
        new_x = np.concatenate([np.real(x), np.imag(x)])
        new_X.append(new_x)
    return np.array(new_X)


def real_to_complex(X):
    """
    WRITEME

    Parameters
    ----------
    X : list of complex vectors

    Notes
    -----
    This function assumes X as 2D
    """
    n = X[0].shape[-1]
    new_X = []
    for i in xrange(len(X)):
        x = X[i]
        real = x[:n/2]
        imag = x[n/2:]
        new_x = real + imag*1.0j
        new_X.append(new_x)
    return np.array(new_X)


def numpy_rfft(X):
    """
    Apply real FFT to X (numpy)

    Parameters
    ----------
    X     : list of lists or ndArrays
    """
    X = np.array([np.fft.rfft(x) for x in X])
    return X


def numpy_irfft(X):
    """
    Apply real inverse FFT to X (numpy)

    Parameters
    ----------
    X     : list of lists or ndArrays
    """
    X = np.array([np.fft.irfft(x) for x in X])
    return X
