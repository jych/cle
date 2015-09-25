import ipdb
import numpy as np
import scipy


class StaticPrepMixin(object):
    """
    Preprocessing mixin for static data
    """
    def normalize(self, X, X_mean=None, X_std=None, axis=0):
        """
        Globally normalize X into zero mean and unit variance

        Parameters
        ----------
        X      : list or ndArray
        X_mean : Scalar
        X_std  : Scalar
        """
        if X_mean is None or X_std is None:
            X_mean = np.array(X).mean(axis=axis)
            X_std = np.array(X).std(axis=axis)
            X = (X - X_mean) / X_std
        else:
            X = (X - X_mean) / X_std
        return (X, X_mean, X_std)

    def global_normalize(self, X, X_mean=None, X_std=None):
        """
        Globally normalize X into zero mean and unit variance

        Parameters
        ----------
        X      : list or ndArray
        X_mean : Scalar
        X_std  : Scalar
        """
        if X_mean is None or X_std is None:
            X_mean = np.array(X).mean()
            X_std = np.array(X).std()
            X = (X - X_mean) / X_std
        else:
            X = (X - X_mean) / X_std
        return (X, X_mean, X_std)

    def standardize(self, X, X_max=None, X_min=None):
        """
        Standardize X such that X \in [0, 1]

        Parameters
        ----------
        X     : list of lists or ndArrays
        X_max : Scalar
        X_min : Scalar
        """
        if X_max is None or X_min is None:
            X_max = np.array(X).max()
            X_min = np.array(X).min()
            X = (X - X_min) / (X_max - X_min)
        else:
            X = (X - X_min) / (X_max - X_min)
        return (X, X_max, X_min)


class SequentialPrepMixin(object):
    """
    Preprocessing mixin for sequential data
    """
    def norm_normalize(self, X, avr_norm=None):
        """
        Unify the norm of each sequence in X

        Parameters
        ----------
        X       : list of lists or ndArrays
        avr_nom : Scalar
        """
        if avr_norm is None:
            avr_norm = 0
            for i in range(len(X)):
                euclidean_norm = np.sqrt(np.square(X[i].sum()))
                X[i] /= euclidean_norm
                avr_norm += euclidean_norm
            avr_norm /= len(X)
        else:
            X = [x[i] / avr_norm for x in X]
        return X, avr_norm

    def global_normalize(self, X, X_mean=None, X_std=None):
        """
        Globally normalize X into zero mean and unit variance

        Parameters
        ----------
        X      : list of lists or ndArrays
        X_mean : Scalar
        X_std  : Scalar

        Notes
        -----
        Compute varaince using the relation
        >>> Var(X) = E[X^2] - E[X]^2
        """
        if X_mean is None or X_std is None:
            X_len = np.array([len(x) for x in X]).sum()
            X_mean = np.array([x.sum() for x in X]).sum() / X_len
            X_sqr = np.array([(x**2).sum() for x in X]).sum() / X_len
            X_std = np.sqrt(X_sqr - X_mean**2)
            X = (X - X_mean) / X_std
        else:
            X = (X - X_mean) / X_std
        return (X, X_mean, X_std)

    def standardize(self, X, X_max=None, X_min=None):
        """
        Standardize X such that X \in [0, 1]

        Parameters
        ----------
        X     : list of lists or ndArrays
        X_max : Scalar
        X_min : Scalar
        """
        if X_max is None or X_min is None:
            X_max = np.array([x.max() for x in X]).max()
            X_min = np.array([x.min() for x in X]).min()
            X = (X - X_min) / (X_max - X_min)
        else:
            X = (X - X_min) / (X_max - X_min)
        return (X, X_max, X_min)

    def numpy_rfft(self, X):
        """
        Apply real FFT to X (numpy)

        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([np.fft.rfft(x) for x in X])
        return X

    def numpy_irfft(self, X):
        """
        Apply real inverse FFT to X (numpy)

        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([np.fft.irfft(x) for x in X])
        return X

    def rfft(self, X):
        """
        Apply real FFT to X (scipy)

        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([scipy.fftpack.rfft(x) for x in X])
        return X

    def irfft(self, X):
        """
        Apply real inverse FFT to X (scipy)

        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([scipy.fftpack.irfft(x) for x in X])
        return X

    def stft(self, X):
        """
        Apply short-time Fourier transform to X

        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([scipy.fft(x) for x in X])
        return X

    def istft(self, X):
        """
        Apply short-time Fourier transform to X

        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([scipy.real(scipy.ifft(x)) for x in X])
        return X

    def fill_zero1D(self, x, pad_len=0, mode='righthand'):
        """
        Given variable lengths sequences,
        pad zeros w.r.t to the maximum
        length sequences and create a
        dense design matrix

        Parameters
        ----------
        X       : list or 1D ndArray
        pad_len : integer
            if 0, we consider that output should be
            a design matrix.
        mode    : string
            Strategy to fill-in the zeros
            'righthand': pad the zeros at the right space
            'lefthand' : pad the zeros at the left space
            'random'   : pad the zeros with randomly
                         chosen left space and right space
        """
        if mode == 'lefthand':
            new_x = np.concatenate([np.zeros((pad_len)), x])
        elif mode == 'righthand':
            new_x = np.concatenate([x, np.zeros((pad_len))])
        elif mode == 'random':
            new_x = np.concatenate(
                [np.zeros((pad_len)), x, np.zeros((pad_len))]
            )
        return new_x

    def fill_zero(self, X, pad_len=0, mode='righthand'):
        """
        Given variable lengths sequences,
        pad zeros w.r.t to the maximum
        length sequences and create a
        dense design matrix

        Parameters
        ----------
        X       : list of ndArrays or lists
        pad_len : integer
            if 0, we consider that output should be
            a design matrix.
        mode    : string
            Strategy to fill-in the zeros
            'righthand': pad the zeros at the right space
            'lefthand' : pad the zeros at the left space
            'random'   : pad the zeros with randomly
                         chosen left space and right space
        """
        if pad_len == 0:
            X_max = np.array([len(x) for x in X]).max()
            new_X = np.zeros((len(X), X_max))
            for i, x in enumerate(X):
                free_ = X_max - len(x)
                if mode == 'lefthand':
                    new_x = np.concatenate([np.zeros((free_)), x], axis=1)
                elif mode == 'righthand':
                    new_x = np.concatenate([x, np.zeros((free_))], axis=1)
                elif mode == 'random':
                    j = np.random.randint(free_)
                    new_x = np.concatenate(
                        [np.zeros((j)), x, np.zeros((free_ - j))],
                        axis=1
                    )
                new_X[i] = new_x
        else:
            new_X = []
            for x in X:
                if mode == 'lefthand':
                    new_x = np.concatenate([np.zeros((pad_len)), x], axis=1)
                elif mode == 'righthand':
                    new_x = np.concatenate([x, np.zeros((pad_len))], axis=1)
                elif mode == 'random':
                    new_x = np.concatenate(
                        [np.zeros((pad_len)), x, np.zeros((pad_len))],
                         axis=1
                    )
                new_X.append(new_x)
        return new_X

    def reverse(self, X):
        """
        Reverse each sequence of X

        Parameters
        ----------
        X       : list of ndArrays or lists
        """
        new_X = []
        for x in X:
            new_X.append(x[::-1])
        return new_X
