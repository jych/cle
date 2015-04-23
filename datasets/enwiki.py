import ipdb
import os
import numpy as np
import theano.tensor as T

from cle.cle.data import TemporalSeries
from cle.cle.data.prep import SequentialPrepMixin
from cle.cle.utils import segment_axis, tolist, totuple


class EnWiki(TemporalSeries, SequentialPrepMixin):
    """
    English Wikipedia dataset batch provider

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, data_mode='chars', batch_size=100, context_len=100, **kwargs):
        self.data_mode = data_mode
        self.batch_size = batch_size
        self.context_len = context_len
        super(EnWiki, self).__init__(**kwargs)

    def load(self, data_path):
        data = np.load(data_path)
        if self.data_mode == 'words':
            if self.name == 'train':
                raw_data = data['train_words']
            elif self.name == 'valid':
                raw_data = data['valid_words']
            elif self.name == 'test':
                raw_data = data['test_words']
            self._max_labels = data['n_words']
        elif self.data_mode == 'chars':
            if self.name == 'train':
                raw_data = data['train_chars']
            elif self.name == 'valid':
                raw_data = data['valid_chars']
            elif self.name == 'test':
                raw_data = data['test_chars']
        chunk_size = len(raw_data) / self.batch_size
        raw_data = segment_axis(raw_data, chunk_size, 0)

        X = []
        y = []
        for i in range(int(np.float((raw_data.shape[1] - 1) /
                           float(self.context_len)))):
            X.extend(raw_data[:, :-1][:, i * self.context_len:(i + 1) * self.context_len,
                                      np.newaxis])
            y.extend(raw_data[:, 1:][:, i * self.context_len:(i + 1) * self.context_len,
                                     np.newaxis])
        X = np.asarray(X)
        y = np.asarray(y)
        return [X, y]

    def theano_vars(self):
        return [T.ftensor3('x'), T.ftensor3('y')]

    def test_theano_vars(self):
        return [T.fmatrix('x')]

    def slices(self, start, end):
        batches = [mat[start:end] for mat in self.data]
        return totuple(batches)


if __name__ == "__main__":
    data_path = '/home/junyoung/data/wikipedia-text/enwiki_char_and_word.npz'
    enwiki = EnWiki(name='train',
                    path=data_path)
    ipdb.set_trace()
