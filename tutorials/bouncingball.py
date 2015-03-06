import ipdb
import numpy as np

from cle.cle.data import Iterator
from cle.cle.graph.net import Net
from cle.cle.layers import InputLayer, InitCell
from cle.cle.layers.cost import MSELayer
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.layers.recurrent import SimpleRecurrent
from cle.cle.train import Training
from cle.cle.train.ext import (
    EpochCount,
    GradientClipping,
    Monitoring,
    Picklize
)
from cle.cle.train.opt import Adam
from cle.cle.utils import unpack
from cle.datasets.bouncing_balls import BouncingBalls


#datapath = '/data/lisatmp3/chungjun/bouncing_balls/bouncing_ball_2balls_16wh_20len_50000cases.npy'
#savepath = '/u/chungjun/repos/cle/saved/'
datapath = '/home/junyoung/data/bouncing_balls/bouncing_ball_2balls_16wh_20len_50000cases.npy'
savepath = '/home/junyoung/repos/cle/saved/'

batchsize = 128
res = 256
debug = 0

trdata = BouncingBalls(name='train',
                       path=datapath)

# Choose the random initialization method
init_W = InitCell('randn')
init_U = InitCell('ortho')
init_b = InitCell('zeros')

# Define nodes: objects
inp, tar = trdata.theano_vars()
# You must use THEANO_FLAGS="compute_test_value=raise" python -m ipdb
if debug:
    inp.tag.test_value = np.zeros((10, batchsize, res), dtype=np.float32)
    tar.tag.test_value = np.zeros((10, batchsize, res), dtype=np.float32)
x = InputLayer(name='x', root=inp, nout=res)
y = InputLayer(name='y', root=tar, nout=res)
# Using skip connections is easy
h1 = SimpleRecurrent(name='h1',
                     parent=[x],
                     batchsize=batchsize,
                     nout=200,
                     unit='tanh',
                     init_W=init_W,
                     init_U=init_U,
                     init_b=init_b)
h2 = SimpleRecurrent(name='h2',
                     parent=[x, h1],
                     batchsize=batchsize,
                     nout=200,
                     unit='tanh',
                     init_W=init_W,
                     init_U=init_U,
                     init_b=init_b)
h3 = SimpleRecurrent(name='h3',
                     parent=[x, h2],
                     batchsize=batchsize,
                     nout=200,
                     unit='tanh',
                     init_W=init_W,
                     init_U=init_U,
                     init_b=init_b)
h4 = FullyConnectedLayer(name='h4',
                         parent=[h1, h2, h3],
                         nout=res,
                         unit='sigmoid',
                         init_W=init_W,
                         init_b=init_b)
cost = MSELayer(name='cost', parent=[h4, y])

nodes = [x, y, h1, h2, h3, h4, cost]
model = Net(nodes=nodes)

cost = unpack(model.build_recurrent_graph(output_args=[cost]))
cost = cost.mean()
cost.name = 'cost'

optimizer = Adam(
    lr=0.01
)

extension = [
    GradientClipping(batchsize=batchsize),
    EpochCount(100),
    Monitoring(freq=100,
               ddout=[cost]),
    Picklize(freq=200,
             path=savepath)
]

mainloop = Training(
    name='toy_bb',
    data=Iterator(trdata, batchsize),
    model=model,
    optimizer=optimizer,
    cost=cost,
    outputs=[cost],
    extension=extension
)
mainloop.run()
