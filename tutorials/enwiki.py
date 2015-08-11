import ipdb
import numpy as np
import theano
import theano.tensor as T

from cle.cle.cost import NllMulInd
from cle.cle.data import Iterator
from cle.cle.models import Model
from cle.cle.layers import InitCell
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.layers.recurrent import GFLSTM
from cle.cle.train import Training
from cle.cle.train.ext import (
    EpochCount,
    GradientClipping,
    Monitoring,
    Picklize
)
from cle.cle.train.opt import Adam
from cle.cle.utils import flatten, sharedX, unpack, OrderedDict
from cle.datasets.enwiki import EnWiki

data_path = '/home/junyoung/data/wikipedia-text/enwiki_char_and_word.npz'
save_path = '/home/junyoung/src/cle/saved/'

batch_size = 100
reset_freq = 100
debug = 0

model = Model()
train_data = EnWiki(name='train',
                    path=data_path)

test_data = EnWiki(name='test',
                   path=data_path)

init_W = InitCell('rand')
init_U = InitCell('ortho')
init_b = InitCell('zeros')

x, y = train_data.theano_vars()
if debug:
    x.tag.test_value = np.zeros((10, batch_size, 1), dtype=np.float32)
    y.tag.test_value = np.zeros((10, batch_size, 1), dtype=np.float32)

h1 = GFLSTM(name='h1',
            parent=['x'],
            parent_dim=[205],
            recurrent=['h2', 'h3'],
            recurrent_dim=[200, 200],
            nout=200,
            unit='tanh',
            init_W=init_W,
            init_U=init_U,
            init_b=init_b)

h2 = GFLSTM(name='h2',
            parent=['x', 'h1'],
            parent_dim=[205, 200],
            recurrent=['h1', 'h3'],
            recurrent_dim=[200, 200],
            nout=200,
            unit='tanh',
            init_W=init_W,
            init_U=init_U,
            init_b=init_b)

h3 = GFLSTM(name='h3',
            parent=['x', 'h2'],
            parent_dim=[205, 200],
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
                             nout=205,
                             unit='softmax',
                             init_W=init_W,
                             init_b=init_b)

nodes = [h1, h2, h3, output]

for node in nodes:
    node.initialize()

params = flatten([node.get_params().values() for node in nodes])

step_count = sharedX(0, name='step_count')
last_h = np.zeros((batch_size, 400), dtype=np.float32)
h1_tm1 = sharedX(last_h, name='h1_tm1')
h2_tm1 = sharedX(last_h, name='h2_tm1')
h3_tm1 = sharedX(last_h, name='h3_tm1')
update_list = [step_count, h1_tm1, h2_tm1, h3_tm1]

step_count = T.switch(T.le(step_count, reset_freq),
                      step_count + 1, 0)

s1_0 = T.switch(T.or_(T.cast(T.eq(step_count, 0), 'int32'),
                      T.cast(T.eq(T.sum(h1_tm1), 0.), 'int32')),
                h1.get_init_state(), h1_tm1)
s2_0 = T.switch(T.or_(T.cast(T.eq(step_count, 0), 'int32'),
                      T.cast(T.eq(T.sum(h2_tm1), 0.), 'int32')),
                h2.get_init_state(), h2_tm1)
s3_0 = T.switch(T.or_(T.cast(T.eq(step_count, 0), 'int32'),
                      T.cast(T.eq(T.sum(h3_tm1), 0.), 'int32')),
                h3.get_init_state(), h3_tm1)


def inner_fn(x_t, h1_tm1, h2_tm1, h3_tm1):

    h1_t = h1.fprop([[x_t], [h1_tm1, h2_tm1, h3_tm1]])
    h2_t = h2.fprop([[x_t, h1_t], [h2_tm1, h1_tm1, h3_tm1]])
    h3_t = h3.fprop([[x_t, h2_t], [h3_tm1, h1_tm1, h2_tm1]])

    return h1_t, h2_t, h3_t

((h1_temp, h2_temp, h3_temp), updates) = theano.scan(fn=inner_fn,
                                                     sequences=[x],
                                                     outputs_info=[s1_0, s2_0, s3_0])

ts, _, _ = y.shape
post_scan_shape = ((ts*batch_size, -1))
h1_in = h1_temp.reshape(post_scan_shape)
h2_in = h2_temp.reshape(post_scan_shape)
h3_in = h3_temp.reshape(post_scan_shape)
y_hat_in = output.fprop([h1_in, h2_in, h3_in])

cost = NllMulInd(y.flatten(), y_hat_in)
cost = cost.mean()
cost.name = 'cost'

model.inputs = [x, y]
model._params = params
model.nodes = nodes
model.set_updates(update_list)

optimizer = Adam(
    lr=0.001
)

extension = [
    GradientClipping(batch_size=batch_size),
    EpochCount(100),
    Monitoring(freq=100,
               ddout=[cost]),
    Picklize(freq=100, path=save_path)
]

mainloop = Training(
    name='toy_enwiki_gflstm',
    data=Iterator(train_data, batch_size),
    model=model,
    optimizer=optimizer,
    cost=cost,
    outputs=[cost],
    extension=extension
)
mainloop.run()
