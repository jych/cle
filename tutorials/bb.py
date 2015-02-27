import ipdb
import numpy as np

from cle.cle.graph.net import Net
from cle.cle.layers import InputLayer, InitCell, MSELayer
from cle.cle.layers.layer import FullyConnectedLayer, SimpleRecurrent
from cle.cle.train import Training
from cle.cle.train.ext import (
    EpochCount,
    GradientClipping,
    Monitoring,
    Picklize
)
from cle.cle.train.opt import Adam
from cle.datasets.bouncing_balls import BouncingBalls


# Mash should be also implemented as data src and using InputLayer!
# Will add it in ver0.2


#datapath = '/data/lisatmp3/chungjun/bouncing_balls/bouncing_ball_2balls_16wh_20len_50000cases.npy'
datapath = '/home/junyoung/data/bouncing_balls/bouncing_ball_2balls_16wh_20len_50000cases.npy'
savepath = '/home/junyoung/repos/cle/saved/'

batch_size = 128

trdata = BouncingBalls(name='train',
                       path=datapath,
                       batch_size=batch_size)

# Choose the random initialization method
init_W, init_U, init_b = InitCell('randn'), InitCell('ortho'), InitCell('zeros')

# Define nodes: objects
inp, tar = trdata.theano_vars()
x = InputLayer(name='inp', root=inp, nout=256)
y = InputLayer(name='tar', target=tar, nout=256)
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
h3 = FullyConnectedLayer(name='h3',
                         parent=[h1],
                         nout=256,
                         unit='sigmoid',
                         init_W=init_W,
                         init_b=init_b)
cost = MSELayer(name='cost', parent=[h2, y])

nodes = [x, y, h1, h2, h3, cost]
model = Net(nodes=nodes)
model.build_recurrent_graph()
#model.build_recurrent_graph(nonseq_args={'a':5, 'b':6})

cost = model.nodes['cost'].out
cost.name = 'cost'

optimizer = Adam(
    lr=0.001
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
