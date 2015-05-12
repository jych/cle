import ipdb
import numpy as np
import theano
import theano.tensor as T

from cle.cle.cost import NllMul
from cle.cle.data import Iterator
from cle.cle.models import Model
from cle.cle.layers import InitCell, OnehotLayer
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
save_path = '/home/junyoung/repos/cle/saved/'

batch_size = 100
reset_freq = 100
debug = 0

model = Model()
trdata = EnWiki(name='train',
                path=data_path)
tedata = EnWiki(name='test',
                path=data_path)

init_W = InitCell('rand')
init_U = InitCell('ortho')
init_b = InitCell('zeros')

x, y = trdata.theano_vars()
if debug:
    x.tag.test_value = np.zeros((10, batch_size, 1), dtype=np.float32)
    y.tag.test_value = np.zeros((10, batch_size, 1), dtype=np.float32)

onehot = OnehotLayer(name='onehot',
                     parent=['x'],
                     nout=205)

h1 = GFLSTM(name='h1',
            parent=['x'],
            parent_dim=[205],
            recurrent=['h2', 'h3'],
            recurrent_dim=[200, 200],
            batch_size=batch_size,
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
            batch_size=batch_size,
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
            batch_size=batch_size,
            nout=200,
            unit='tanh',
            init_W=init_W,
            init_U=init_U,
            init_b=init_b)

h4 = FullyConnectedLayer(name='h4',
                         parent=['h1', 'h2', 'h3'],
                         parent_dim=[200, 200, 200],
                         nout=205,
                         unit='softmax',
                         init_W=init_W,
                         init_b=init_b)

nodes = [onehot, h1, h2, h3, h4]

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

h1_init_state = T.switch(
                T.or_(T.cast(T.eq(step_count, 0), 'int32'),
                      T.cast(T.eq(T.sum(h1_tm1), 0.), 'int32')),
                h1.get_init_state(), h1_tm1
            )
h2_init_state = T.switch(
                T.or_(T.cast(T.eq(step_count, 0), 'int32'),
                      T.cast(T.eq(T.sum(h2_tm1), 0.), 'int32')),
                h2.get_init_state(), h2_tm1
            )
h3_init_state = T.switch(
                T.or_(T.cast(T.eq(step_count, 0), 'int32'),
                      T.cast(T.eq(T.sum(h3_tm1), 0.), 'int32')),
                h3.get_init_state(), h3_tm1
            )


def inner_fn(i_t, h1_tm1, h2_tm1, h3_tm1):

    x_t = onehot.fprop([i_t])
    h1_t = h1.fprop([[x_t], [h1_tm1, h2_tm1, h3_tm1]])
    h2_t = h2.fprop([[x_t, h1_t], [h2_tm1, h1_tm1, h3_tm1]])
    h3_t = h3.fprop([[x_t, h2_t], [h3_tm1, h1_tm1, h2_tm1]])
    y_hat = h4.fprop([h1_t, h2_t, h3_t])

    return h1_t, h2_t, h3_t, y_hat

((h1_t, h2_t, h3_t, y_hat),
 updates) =\
    theano.scan(fn=inner_fn,
                sequences=[x],
                outputs_info=[h1_init_state,
                              h2_init_state,
                              h3_init_state,
                              None])

reshaped_y = y.reshape((y.shape[0]*y.shape[1], -1))
reshaped_y = onehot.fprop([reshaped_y])
reshaped_y_hat = y_hat.reshape((y_hat.shape[0]*y_hat.shape[1], -1))

cost = NllMul(reshaped_y, reshaped_y_hat)
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
    Picklize(freq=10000,
             path=save_path)
]

mainloop = Training(
    name='toy_enwiki_gflstm',
    data=Iterator(trdata, batch_size),
    model=model,
    optimizer=optimizer,
    cost=cost,
    outputs=[cost],
    extension=extension
)
mainloop.run()
