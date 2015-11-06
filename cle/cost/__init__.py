import ipdb
import numpy as np
import theano.tensor as T

from cle.cle.utils.op import logsumexp


def NllBin(y, y_hat):
    """
    Binary cross-entropy

    Parameters
    ----------
    .. todo::
    """
    nll = T.nnet.binary_crossentropy(y_hat, y).sum(axis=-1)
    return nll


def NllMul(y, y_hat):
    """
    Multi cross-entropy

    Parameters
    ----------
    .. todo::
    """
    ll = (y * T.log(y_hat)).sum(axis=-1)
    nll = -ll
    return nll


def NllMulInd(y, y_hat):
    """
    Multi cross-entropy
    Efficient implementation using the indices in y

    Credit assignment:
    This code is brought from: https://github.com/lisa-lab/pylearn2

    Parameters
    ----------
    .. todo::
    """
    log_prob = T.log(y_hat)
    flat_log_prob = log_prob.flatten()
    flat_y = y.flatten()
    flat_indices = flat_y + T.arange(y.shape[0]) * log_prob.shape[1]
    ll = flat_log_prob[T.cast(flat_indices, 'int64')]
    nll = -ll
    return nll


def MSE(y, y_hat, use_sum=1):
    """
    Mean squared error

    Parameters
    ----------
    .. todo::
    """
    if use_sum:
        mse = T.sum(T.sqr(y - y_hat), axis=-1)
    else:
        mse = T.mean(T.sqr(y - y_hat), axis=-1)
    return mse


def Laplace(y, mu, sig):
    """
    Gaussian negative log-likelihood

    Parameters
    ----------
    y   : TensorVariable
    mu  : FullyConnected (Linear)
    sig : FullyConnected (Softplus)
    """
    nll = T.sum(abs(y - mu) / sig + T.log(sig) + T.log(2), axis=-1)
    return nll


def Gaussian(y, mu, sig):
    """
    Gaussian negative log-likelihood

    Parameters
    ----------
    y   : TensorVariable
    mu  : FullyConnected (Linear)
    sig : FullyConnected (Softplus)
    """
    nll = 0.5 * T.sum(T.sqr(y - mu) / sig**2 + 2 * T.log(sig) +
                      T.log(2 * np.pi), axis=-1)
    return nll


def GMM(y, mu, sig, coeff):
    """
    Gaussian mixture model negative log-likelihood

    Parameters
    ----------
    y     : TensorVariable
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    coeff : FullyConnected (Softmax)
    """
    y = y.dimshuffle(0, 1, 'x')
    mu = mu.reshape((mu.shape[0],
                     mu.shape[1]/coeff.shape[-1],
                     coeff.shape[-1]))
    sig = sig.reshape((sig.shape[0],
                       sig.shape[1]/coeff.shape[-1],
                       coeff.shape[-1]))
    inner = -0.5 * T.sum(T.sqr(y - mu) / sig**2 + 2 * T.log(sig) +
                         T.log(2 * np.pi), axis=1)
    nll = -logsumexp(T.log(coeff) + inner, axis=1)

    return nll


def BiGauss(y, mu, sig, corr, binary):
    """
    Gaussian mixture model negative log-likelihood
    Parameters
    ----------
    y     : TensorVariable
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    """
    mu_1 = mu[:, 0].reshape((-1, 1))
    mu_2 = mu[:, 1].reshape((-1, 1))

    sig_1 = sig[:, 0].reshape((-1, 1))
    sig_2 = sig[:, 1].reshape((-1, 1))

    y0 = y[:, 0].reshape((-1, 1))
    y1 = y[:, 1].reshape((-1, 1))
    y2 = y[:, 2].reshape((-1, 1))
    corr = corr.reshape((-1, 1))

    c_b =  T.sum(T.xlogx.xlogy0(y0, binary) +
                T.xlogx.xlogy0(1 - y0, 1 - binary), axis=1)

    inner1 =  ((0.5*T.log(1-corr**2)) +
               T.log(sig_1) + T.log(sig_2) + T.log(2 * np.pi))

    z = (((y1 - mu_1) / sig_1)**2 + ((y2 - mu_2) / sig_2)**2 -
         (2. * (corr * (y1 - mu_1) * (y2 - mu_2)) / (sig_1 * sig_2)))

    inner2 = 0.5 * (1. / (1. - corr**2))
    cost = - (inner1 + (inner2 * z))

    nll = -T.sum(cost ,axis=1) - c_b

    return nll


def BiGMM(y, mu, sig, coeff, corr, binary):
    """
    Bivariate Gaussian mixture model negative log-likelihood
    Parameters
    ----------
    y     : TensorVariable
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    coeff : FullyConnected (Softmax)
    corr  : FullyConnected (Tanh)
    binary: FullyConnected (Sigmoid)
    """
    y = y.dimshuffle(0, 1, 'x')

    mu = mu.reshape((mu.shape[0],
                     mu.shape[1] / coeff.shape[-1],
                     coeff.shape[-1]))

    mu_1 = mu[:, 0, :]
    mu_2 = mu[:, 1, :]

    sig = sig.reshape((sig.shape[0],
                       sig.shape[1] / coeff.shape[-1],
                       coeff.shape[-1]))

    sig_1 = sig[:, 0, :]
    sig_2 = sig[:, 1, :]

    c_b = T.sum(T.xlogx.xlogy0(y[:, 0, :], binary) +
                T.xlogx.xlogy0(1 - y[:, 0, :], 1 - binary), axis=1)

    inner1 = (0.5 * T.log(1 - corr ** 2) +
              T.log(sig_1) + T.log(sig_2) + T.log(2 * np.pi))

    z = (((y[:, 1, :] - mu_1) / sig_1)**2 + ((y[:, 2, :] - mu_2) / sig_2)**2 -
         (2. * (corr * (y[:, 1, :] - mu_1) * (y[:, 2, :] - mu_2)) / (sig_1 * sig_2)))

    inner2 = 0.5 * (1. / (1. - corr**2))
    cost = -(inner1 + (inner2 * z))

    nll = -logsumexp(T.log(coeff) + cost, axis=1) - c_b

    return nll


def KLGaussianStdGaussian(mu, sig):
    """
    Re-parameterized formula for KL
    between Gaussian predicted by encoder and standardized Gaussian dist.

    Parameters
    ----------
    mu  : FullyConnected (Linear)
    sig : FullyConnected (Softplus)
    """
    kl = T.sum(0.5 * (-2 * T.log(sig) + mu**2 + sig**2 - 1), axis=-1)

    return kl


def KLGaussianGaussian(mu1, sig1, mu2, sig2, keep_dims=0):
    """
    Re-parameterized formula for KL
    between Gaussian predicted by encoder and Gaussian dist.

    Parameters
    ----------
    mu1  : FullyConnected (Linear)
    sig1 : FullyConnected (Softplus)
    mu2  : FullyConnected (Linear)
    sig2 : FullyConnected (Softplus)
    """
    if keep_dims:
        kl = 0.5 * (2 * T.log(sig2) - 2 * T.log(sig1) +
                    (sig1**2 + (mu1 - mu2)**2) / sig2**2 - 1)
    else:
        kl = T.sum(0.5 * (2 * T.log(sig2) - 2 * T.log(sig1) +
                   (sig1**2 + (mu1 - mu2)**2) /
                   sig2**2 - 1), axis=-1)

    return kl


def grbm_free_energy(v, W, X):
    """
    Gaussian restricted Boltzmann machine free energy

    Parameters
    ----------
    to do::
    """
    bias_term = 0.5*(((v - X[1])/X[2])**2).sum(axis=1)
    hidden_term = T.log(1 + T.exp(T.dot(v/X[2], W) + X[0])).sum(axis=1)
    FE = bias_term -hidden_term

    return FE
