import ipdb
import theano.tensor as T

from cle.cle.utils.op import logsumexp


def NllBin(y, y_hat):
    """
    Binary cross-entropy

    Parameters
    ----------
    .. todo::
    """
    nll = -T.sum(y * T.log(y_hat) + (1 - y) * T.log(1 - y_hat),
                 axis=-1)
    return nll


def NllMul(y, y_hat):
    """
    Multi cross-entropy

    Parameters
    ----------
    .. todo::
    """
    nll =  -T.sum(y * T.log(y_hat), axis=-1)
    return nll


def MSE(y, y_hat):
    """
    Mean squared error

    Parameters
    ----------
    .. todo::
    """
    mse =  T.sum(T.sqr(y - y_hat), axis=-1)
    return mse


def Gaussian(y, mu, logvar):
    """
    Gaussian negative log-likelihood

    Parameters
    ----------
    y      : TensorVariable
    mu     : FullyConnected (Linear)
    logvar : FullyConnected (Linear)
    """
    nll = 0.5 * T.sum(T.sqr(y - mu) * T.exp(-logvar) + logvar, axis=1)
    return nll


def GMM(y, mu, logvar, coeff):
    """
    Gaussian mixture model negative log-likelihood

    Parameters
    ----------
    y      : TensorVariable
    mu     : FullyConnected (Linear)
    logvar : FullyConnected (Linear)
    coeff  : FullyConnected (Softmax)
    """
    y = y.dimshuffle(0, 1, 'x')
    mu = mu.reshape((mu.shape[0],
                     mu.shape[1]/coeff.shape[-1],
                     coeff.shape[-1]))
    logvar = logvar.reshape((logvar.shape[0],
                             logvar.shape[1]/coeff.shape[-1],
                             coeff.shape[-1]))
    nll = 0.5 * T.sum(T.sqr(y - mu) * T.exp(-logvar) + logvar, axis=1)
    nll = logsumexp(T.log(coeff) + nll, axis=-1)
    return nll
