import ipdb
import numpy as np
import theano

from cle.cle.cost import MSE
from cle.cle.data import Iterator
from cle.cle.models import Model
from cle.cle.layers import InitCell
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.layers.recurrent import GFGRU
from cle.cle.train import Training
from cle.cle.train.ext import (
    EpochCount,
    GradientClipping,
    Monitoring,
    Picklize
)
from cle.cle.train.opt import Adam
from cle.cle.utils import init_tparams, OrderedDict
from cle.datasets.bouncing_balls import BouncingBalls


data_path = '/data/lisatmp3/chungjun/bouncing_balls/bouncing_ball_2balls_16wh_20len_50000cases.npy'
save_path = '/u/chungjun/repos/cle/saved/'

batch_size = 128
frame_size = 256
debug = 1

model = Model()
train_data = BouncingBalls(name='train',
                           path=data_path)

valid_data = BouncingBalls(name='valid',
                           path=data_path)

x, y = train_data.theano_vars()

if debug:
    x.tag.test_value = np.zeros((10, batch_size, frame_size), dtype=np.float32)
    y.tag.test_value = np.zeros((10, batch_size, frame_size), dtype=np.float32)

init_W = InitCell('randn')
init_U = InitCell('ortho')
init_b = InitCell('zeros')

h1 = GFGRU(name='h1',
           parent=['x'],
           parent_dim=[frame_size],
           recurrent=['h2', 'h3'],
           recurrent_dim=[200, 200],
           nout=200,
           unit='tanh',
           init_W=init_W,
           init_U=init_U,
           init_b=init_b)

h2 = GFGRU(name='h2',
           parent=['h1'],
           parent_dim=[200],
           recurrent=['h1', 'h3'],
           recurrent_dim=[200, 200],
           nout=200,
           unit='tanh',
           init_W=init_W,
           init_U=init_U,
           init_b=init_b)

h3 = GFGRU(name='h3',
           parent=['h2'],
           parent_dim=[200],
           recurrent=['h1', 'h2'],
           recurrent_dim=[200, 200],
           nout=200,
           unit='tanh',
           init_W=init_W,
           init_U=init_U,
           init_b=init_b)

output = FullyConnectedLayer(name='output',
                             parent=['h1', 'h2', 'h3'],
                             parent_dim=[200, 200, 200],
                             nout=frame_size,
                             unit='sigmoid',
                             init_W=init_W,
                             init_b=init_b)

nodes = [h1, h2, h3, output]

params = OrderedDict()
for node in nodes:
    params.update(node.initialize())
params = init_tparams(params)

s1_0 = h1.get_init_state(batch_size)
s2_0 = h2.get_init_state(batch_size)
s3_0 = h3.get_init_state(batch_size)


def inner_fn(x_t, s1_tm1, s2_tm1, s3_tm1):

    s1_t = h1.fprop([[x_t], [s1_tm1, s2_tm1, s3_tm1]], params)
    s2_t = h2.fprop([[s1_t], [s2_tm1, s1_tm1, s3_tm1]], params)
    s3_t = h3.fprop([[s2_t], [s3_tm1], s1_tm1, s2_tm1], params)
    y_hat_t = output.fprop([s1_t, s2_t, s3_t], params)

    return s1_t, s2_t, s3_t, y_hat_t

((h1_temp, h2_temp, h3_temp, y_hat_temp), updates) =\
    theano.scan(fn=inner_fn,
                sequences=[x],
                outputs_info=[s1_0, s2_0, s3_0, None])

mse = MSE(y, y_hat_temp)
mse = mse.mean()
mse.name = 'mse'

model.inputs = [x, y]
model.params = params
model.nodes = nodes

optimizer = Adam(
    lr=0.001
)

extension = [
    GradientClipping(batch_size=batch_size),
    EpochCount(100),
    Monitoring(freq=100,
               ddout=[mse]),
    Picklize(freq=200, path=save_path)
]

mainloop = Training(
    name='toy_bb_gfgru',
    data=Iterator(trdata, batch_size),
    model=model,
    optimizer=optimizer,
    cost=mse,
    outputs=[mse],
    extension=extension
)
mainloop.run()
