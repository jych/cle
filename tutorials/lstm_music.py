import ipdb
import numpy as np

from cle.cle.graph.net import Net
from cle.cle.layers import InputLayer, InitCell, MSELayer
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.layers.recurrent import LSTM
from cle.cle.train import Training
from cle.cle.train.ext import (
    EpochCount,
    GradientClipping,
    Monitoring,
    Picklize
)
from cle.cle.train.opt import Adam
from cle.cle.utils import unpack
from cle.datasets.music import Music


#datapath = '/data/lisatmp3/chungjun/bouncing_balls/bouncing_ball_2balls_16wh_20len_50000cases.npy'
#savepath = '/u/chungjun/repos/cle/saved/'
datapath = '/home/junyoung/data/music/MuseData.pickle'
savepath = '/home/junyoung/repos/cle/saved/'

batchsize = 100
trdata = Music(name='train',
               path=datapath,
               nlabel=105,
               batchsize=batchsize)

# Choose the random initialization method
init_W, init_U, init_b = InitCell('randn'), InitCell('ortho'), InitCell('zeros')

# Define nodes: objects
inp, tar, mask = trdata.theano_vars()
x = InputLayer(name='x', root=inp, nout=256)
y = InputLayer(name='y', root=tar, nout=256)
mask = InputLayer(name='mask', root=mask, nout=1)
ipdb.set_trace()
# Using skip connections is easy
h1 = LSTM(name='h1',
          parent=[x],
          batchsize=batchsize,
          nout=200,
          unit='tanh',
          init_W=init_W,
          init_U=init_U,
          init_b=init_b)
h2 = LSTM(name='h2',
          parent=[x, h1],
          batchsize=batchsize,
          nout=200,
          unit='tanh',
          init_W=init_W,
          init_U=init_U,
          init_b=init_b)
h3 = LSTM(name='h3',
          parent=[x, h2],
          batchsize=batchsize,
          nout=200,
          unit='tanh',
          init_W=init_W,
          init_U=init_U,
          init_b=init_b)
h4 = FullyConnectedLayer(name='h4',
                         parent=[h1, h2, h3],
                         nout=256,
                         unit='sigmoid',
                         init_W=init_W,
                         init_b=init_b)
masked_y = MaskLayer(name='masked_y', parent=[y, mask])
masked_y_hat = MaskLayer(name='masked_y_hat', parent[h4, mask])
cost = MSELayer(name='cost', parent=[masked_y_hat, masked_y])
nodes = [x, y, h1, h2, h3, h4, cost, mask]
model = Net(nodes=nodes)

# You can either use dict or list
cost = unpack(model.build_recurrent_graph(output_args=[cost]))
cost = cost[-1]
cost.name = 'cost'

optimizer = Adam(
    lr=0.001
)

extension = [
    GradientClipping(batchsize),
    EpochCount(100),
    Monitoring(freq=100,
               ddout=[cost]),
    Picklize(freq=1,
             path=savepath)
]

mainloop = Training(
    name='toy_music',
    data=trdata,
    model=model,
    optimizer=optimizer,
    cost=cost,
    outputs=[cost],
    extension=extension
)
mainloop.run()
