import ipdb
import theano.tensor as T
pi = sharedX(np.pi)


def NllBin(y, y_hat):
    """
    Binary cross-entropy

    Parameters
    ----------
    todo..
    """
    nll = -T.sum(y * T.log(y_hat) + (1-y) * T.log(1-y_hat), axis=-1)
    return nll.mean()


def NllMul(y, y_hat):
    """
    Multi cross-entropy

    Parameters
    ----------
    todo..
    """
    nll =  -T.sum(y * T.log(y_hat), axis=-1)
    return nll.mean()


def MSE(y, y_hat):
    """
    Mean squared error

    Parameters
    ----------
    todo..
    """
    mse =  T.sum(T.sqr(y - y_hat), axis=-1)
    return mse.mean()


def Gaussian(y, mu, logvar):
    """
    Gaussian negative log-likelihood

    Parameters
    ----------
    todo..
    """
    ll = T.sum(T.sqr(y - mu) * T.exp(-logvar) +
                 logvar + T.log(2 * pi), axis=1)
    ll *= 0.5
    nll = -ll
    return nll.mean()


def MOG(y, mu, logvar):
    """
    Mixture of Gaussian negative log-likelihood

    Parameters
    ----------
    todo..
    """
    z = (y - mu)
    ll = T.sum(T.sqr(z) * T.exp(-logvar) +
                 logvar + T.log(2 * pi), axis=1)
    ll *= 0.5
    nll = -ll
    return nll.mean()
