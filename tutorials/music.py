import ipdb
import numpy as np

from cle.cle.data import Iterator
from cle.cle.cost import NllBin
from cle.cle.graph.net import Net
from cle.cle.layers import InputLayer, InitCell
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
from cle.datasets.music import Music


datapath = '/data/lisa/data/music/MuseData.pickle'
savepath = '/u/chungjun/repos/cle/saved/'
#datapath = '/home/junyoung/data/music/MuseData.pickle'
#savepath = '/home/junyoung/repos/cle/saved/'

batch_size = 10
nlabel = 105
debug = 1

trdata = Music(name='train',
               path=datapath,
               nlabel=nlabel)
valdata = Music(name='valid',
                path=datapath,
                nlabel=nlabel)

# Choose the random initialization method
init_W = InitCell('randn')
init_U = InitCell('ortho')
init_b = InitCell('zeros')

inp, y, mask = trdata.theano_vars()
# You must use THEANO_FLAGS="compute_test_value=raise" python -m ipdb
if debug:
    inp.tag.test_value = np.zeros((10, batch_size, nlabel), dtype=np.float32)
    y.tag.test_value = np.zeros((10, batch_size, nlabel), dtype=np.float32)
    mask.tag.test_value = np.ones((10, batch_size), dtype=np.float32)
x = InputLayer(name='x', root=inp, nout=nlabel)
h1 = LSTM(name='h1',
          parent=[x],
          batch_size=batch_size,
          nout=50,
          unit='tanh',
          init_W=init_W,
          init_U=init_U,
          init_b=init_b)
h2 = LSTM(name='h2',
          parent=[x, h1],
          batch_size=batch_size,
          nout=50,
          unit='tanh',
          init_W=init_W,
          init_U=init_U,
          init_b=init_b)
h3 = LSTM(name='h3',
          parent=[x, h2],
          batch_size=batch_size,
          nout=50,
          unit='tanh',
          init_W=init_W,
          init_U=init_U,
          init_b=init_b)
h4 = FullyConnectedLayer(name='h4',
                         parent=[h1, h2, h3],
                         nout=nlabel,
                         unit='sigmoid',
                         init_W=init_W,
                         init_b=init_b)
nodes = [x, h1, h2, h3, h4]
model = Net(nodes=nodes)

y_hat = model.build_recurrent_graph(output_args=[h4])[0]
masked_y = y[mask.nonzero()]
masked_y_hat = y_hat[mask.nonzero()]
cost = NllBin(masked_y, masked_y_hat).sum()
nll = NllBin(masked_y, masked_y_hat).mean()
cost.name = 'cost'
nll.name = 'nll'
model.inputs += [y, mask]

optimizer = RMSProp(
    lr=0.0001,
    mom=0.95
)

extension = [
    GradientClipping(batch_size=batch_size),
    EpochCount(100),
    Monitoring(freq=10,
               ddout=[cost, nll],
               data=[Iterator(valdata, batch_size)]),
    Picklize(freq=5,
             path=savepath)
]

mainloop = Training(
    name='toy_music',
    data=Iterator(trdata, batch_size),
    model=model,
    optimizer=optimizer,
    cost=cost,
    outputs=[cost],
    extension=extension
)
mainloop.run()
