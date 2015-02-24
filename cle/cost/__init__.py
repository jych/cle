import theano.tensor as T


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
