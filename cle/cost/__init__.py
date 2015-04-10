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
    nll = T.nnet.binary_crossentropy(y_hat, y).sum(axis=1)
    return nll


def NllMul(y, y_hat):
    """
    Multi cross-entropy

    Parameters
    ----------
    .. todo::
    """
    nll = -T.sum(y * T.log(y_hat), axis=-1)
    return nll


def MSE(y, y_hat):
    """
    Mean squared error

    Parameters
    ----------
    .. todo::
    """
    mse = T.sum(T.sqr(y - y_hat), axis=-1)
    return mse


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
                      T.log(2 * np.pi), axis=1)
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


def KLGaussianGaussian(mu1, sig1, mu2, sig2):
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
    kl = T.sum(0.5 * (2 * T.log(sig2) - 2 * T.log(sig1) + (sig1**2 + (mu1 - mu2)**2) /
               sig2**2 - 1), axis=-1)
    return kl


def KLGaussianGaussianKeepDims(mu1, sig1, mu2, sig2):
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
    kl = 0.5 * (2 * T.log(sig2) - 2 * T.log(sig1) + (sig1**2 + (mu1 - mu2)**2) / sig2**2 - 1)
    return kl   
