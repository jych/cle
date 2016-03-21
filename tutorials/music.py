import ipdb
import numpy as np
import theano

from cle.cle.cost import NllBin
from cle.cle.data import Iterator
from cle.cle.models import Model
from cle.cle.layers import InitCell
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.layers.recurrent import LSTM
from cle.cle.train import Training
from cle.cle.train.ext import (
    EpochCount,
    GradientClipping,
    Monitoring,
    Picklize
)
from cle.cle.train.opt import RMSProp
from cle.cle.utils import init_tparams, sharedX
from cle.cle.utils.compat import OrderedDict
from cle.datasets.music import Music

data_path = '/home/junyoung/data/music/MuseData.pickle'
save_path = '/home/junyoung/repos/cle/saved/'

batch_size = 10
nlabel = 105
debug = 1

model = Model()
train_data = Music(name='train',
                   path=data_path,
                   nlabel=nlabel)

valid_data = Music(name='valid',
                   path=data_path,
                   nlabel=nlabel)

# Choose the random initialization method
init_W = InitCell('randn')
init_U = InitCell('ortho')
init_b = InitCell('zeros')

x, y, mask = train_data.theano_vars()
# You must use THEANO_FLAGS="compute_test_value=raise" python -m ipdb
if debug:
    x.tag.test_value = np.zeros((10, batch_size, nlabel), dtype=np.float32)
    y.tag.test_value = np.zeros((10, batch_size, nlabel), dtype=np.float32)
    mask.tag.test_value = np.ones((10, batch_size), dtype=np.float32)

h1 = LSTM(name='h1',
          parent=['x'],
          parent_dim=[105],
          nout=50,
          unit='tanh',
          init_W=init_W,
          init_U=init_U,
          init_b=init_b)

h2 = LSTM(name='h2',
          parent=['h1'],
          parent_dim=[50],
          nout=50,
          unit='tanh',
          init_W=init_W,
          init_U=init_U,
          init_b=init_b)

h3 = LSTM(name='h3',
          parent=['h2'],
          parent_dim=[50],
          nout=50,
          unit='tanh',
          init_W=init_W,
          init_U=init_U,
          init_b=init_b)

output = FullyConnectedLayer(name='output',
                             parent=['h1', 'h2', 'h3'],
                             parent_dim=[50, 50, 50],
                             nout=nlabel,
                             unit='sigmoid',
                             init_W=init_W,
                             init_b=init_b)

nodes = [h1, h2, h3, output]

params = OrderedDict()

for node in nodes:
    if node.initialize() is not None:
        params.update(node.initialize())

params = init_tparams(params)

s1_0 = h1.get_init_state(batch_size)
s2_0 = h2.get_init_state(batch_size)
s3_0 = h3.get_init_state(batch_size)


def inner_fn(x_t, s1_tm1, s2_tm1, s3_tm1):

    h1_t = h1.fprop([[x_t], [s1_tm1]], params)
    h2_t = h2.fprop([[h1_t], [s2_tm1]], params)
    h3_t = h3.fprop([[h2_t], [s2_tm1]], params)
    output_t = output.fprop([h1_t, h2_t, h3_t], params)

    return h1_t, h2_t, h3_t, output_t

((h1_temp, h2_temp, h3_temp, y_hat_temp), updates) =\
    theano.scan(fn=inner_fn,
                sequences=[x],
                outputs_info=[s1_0, s2_0, s3_0, None])

ts, _, _ = y_hat_temp.shape
y_hat_in = y_hat_temp.reshape((ts*batch_size, -1))
y_in = y.reshape((ts*batch_size, -1))
cost = NllBin(y_in, y_hat_in)
cost_temp = cost.reshape((ts, batch_size))
cost = cost_temp * mask
nll = cost.sum() / mask.sum()
cost = cost.sum(axis=0).mean()
cost.name = 'cost'
nll.name = 'nll'

model.inputs = [x, y, mask]
model.params = params
model.nodes = nodes

optimizer = RMSProp(
    lr=0.0001,
    mom=0.95
)

extension = [
    GradientClipping(batch_size=batch_size),
    EpochCount(100),
    Monitoring(freq=10,
               ddout=[cost, nll],
               data=[Iterator(valid_data, batch_size)]),
    Picklize(freq=10, path=save_path)
]

mainloop = Training(
    name='toy_music',
    data=Iterator(train_data, batch_size),
    model=model,
    optimizer=optimizer,
    cost=cost,
    outputs=[cost],
    extension=extension
)
mainloop.run()
