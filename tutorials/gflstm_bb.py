import ipdb
import numpy as np

from cle.cle.graph.net import Net
from cle.cle.layers import InputLayer, InitCell, MSELayer
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
from cle.cle.util import unpack
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

# Using skip connections is easy
h1 = GFLSTM(name='h1',
            parent=[x],
            batch_size=batch_size,
            nout=200,
            unit='tanh',
            init_W=init_W,
            init_U=init_U,
            init_b=init_b)
h2 = GFLSTM(name='h2',
            parent=[x, h1],
            recurrent=[h1],
            batch_size=batch_size,
            nout=200,
            unit='tanh',
            init_W=init_W,
            init_U=init_U,
            init_b=init_b)
h3 = GFLSTM(name='h3',
            parent=[x, h2],
            recurrent=[h1, h2],
            batch_size=batch_size,
            nout=200,
            unit='tanh',
            init_W=init_W,
            init_U=init_U,
            init_b=init_b)
h2.recurrent.append(h3)
h1.recurrent.append(h2)
h1.recurrent.append(h3)
h4 = FullyConnectedLayer(name='h4',
                         parent=[h1, h2, h3],
                         nout=256,
                         unit='sigmoid',
                         init_W=init_W,
                         init_b=init_b)
cost = MSELayer(name='cost', parent=[h4, y])

nodes = [x, y, h1, h2, h3, h4, cost]
model = Net(nodes=nodes)

# You can either use dict or list
#cost = model.build_recurrent_graph(output_args={'cost': cost})
cost = unpack(model.build_recurrent_graph(output_args=[cost]))
cost = cost[-1]
cost.name = 'cost'

optimizer = Adam(
    lr=0.001
)

extension = [
    GradientClipping(batch_size),
    EpochCount(40),
    Monitoring(freq=100,
               ddout=[cost]),
    Picklize(freq=200,
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
