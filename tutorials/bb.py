import ipdb
import numpy as np

from cle.cle.graph.net import Net
from cle.cle.layers import InputLayer, InitCell, MSELayer
from cle.cle.layers.layer import FullyConnectedLayer, SimpleRecurrent, LSTM
from cle.cle.train import Training
from cle.cle.train.ext import (
    EpochCount,
    GradientClipping,
    Monitoring,
    Picklize
)
from cle.cle.train.opt import Adam
from cle.datasets.bouncing_balls import BouncingBalls


# Mask should be also implemented as data src and using InputLayer!
# Will add it in ver0.2


#datapath = '/data/lisatmp3/chungjun/bouncing_balls/bouncing_ball_2balls_16wh_20len_50000cases.npy'
#savepath = '/u/chungjun/repos/cle/saved/'
datapath = '/home/junyoung/data/bouncing_balls/bouncing_ball_2balls_16wh_20len_50000cases.npy'
savepath = '/home/junyoung/repos/cle/saved/'

batch_size = 100

trdata = BouncingBalls(name='train',
                       path=datapath,
                       batch_size=batch_size)

# Choose the random initialization method
init_W, init_U, init_b = InitCell('randn'), InitCell('ortho'), InitCell('zeros')

# Define nodes: objects
inp, tar = trdata.theano_vars()
x = InputLayer(name='x', root=inp, nout=256)
y = InputLayer(name='y', root=tar, nout=256)

# Try simple RNN (tanh)
if 0:
    h1 = SimpleRecurrent(name='h1',
                         parent=[x],
                         batch_size=batch_size,
                         nout=200,
                         unit='tanh',
                         init_W=init_W,
                         init_U=init_U,
                         init_b=init_b)
    h2 = SimpleRecurrent(name='h2',
                        parent=[h1],
                        batch_size=batch_size,
                        nout=200,
                        unit='tanh',
                        init_W=init_W,
                        init_U=init_U,
                        init_b=init_b)
# Try LSTM
if 1:
    h1 = LSTM(name='h1',
              parent=[x],
              batch_size=batch_size,
              nout=200,
              unit='tanh',
              init_W=init_W,
              init_U=init_U,
              init_b=init_b)
    h2 = LSTM(name='h2',
              parent=[h1],
              batch_size=batch_size,
              nout=200,
              unit='tanh',
              init_W=init_W,
              init_U=init_U,
              init_b=init_b)
h3 = FullyConnectedLayer(name='h3',
                         parent=[h2],
                         nout=256,
                         unit='sigmoid',
                         init_W=init_W,
                         init_b=init_b)
cost = MSELayer(name='cost', parent=[h3, y])

nodes = [x, y, h1, h2, h3, cost]
model = Net(nodes=nodes)

# You can either use dict or list
#cost = model.build_recurrent_graph(output_args={'cost': cost})[0][-1]
cost = model.build_recurrent_graph(output_args=[cost])[0][-1]
cost.name = 'cost'

optimizer = Adam(
    lr=0.01
)

extension = [
    GradientClipping(batch_size),
    EpochCount(40),
    Monitoring(freq=100,
               ddout=[cost]),
    Picklize(freq=10,
             path=savepath)
]

mainloop = Training(
    name='toy_bb',
    data=trdata,
    model=model,
    optimizer=optimizer,
    cost=cost,
    outputs=[cost],
    extension=extension
)
mainloop.run()
