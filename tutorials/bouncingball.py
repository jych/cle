import ipdb
import numpy as np

from cle.cle.graph.net import Net
from cle.cle.data import Iterator
from cle.cle.models import Model
from cle.cle.layers import InitCell
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
from cle.cle.utils import unpack, OrderedDict
from cle.datasets.bouncing_balls import BouncingBalls


#data_path = '/data/lisatmp3/chungjun/bouncing_balls/bouncing_ball_2balls_16wh_20len_50000cases.npy'
#save_path = '/u/chungjun/repos/cle/saved/'
data_path = '/home/junyoung/data/bouncing_balls/bouncing_ball_2balls_16wh_20len_50000cases.npy'
save_path = '/home/junyoung/repos/cle/saved/'

batch_size = 128
res = 256
debug = 0

model = Model()
trdata = BouncingBalls(name='train',
                       path=data_path)

# Choose the random initialization method
init_W = InitCell('randn')
init_U = InitCell('ortho')
init_b = InitCell('zeros')

# Define nodes: objects
model.inputs = trdata.theano_vars()
x, y = model.inputs
# You must use THEANO_FLAGS="compute_test_value=raise" python -m ipdb
if debug:
    x.tag.test_value = np.zeros((10, batch_size, res), dtype=np.float32)
    y.tag.test_value = np.zeros((10, batch_size, res), dtype=np.float32)

inputs = [x, y]
inputs_dim = {'x':256, 'y':256}

# Using skip connections is easy
h1 = SimpleRecurrent(name='h1',
                     parent=['x'],
                     batch_size=batch_size,
                     nout=200,
                     unit='tanh',
                     init_W=init_W,
                     init_U=init_U,
                     init_b=init_b)

h2 = SimpleRecurrent(name='h2',
                     parent=['x', 'h1'],
                     batch_size=batch_size,
                     nout=200,
                     unit='tanh',
                     init_W=init_W,
                     init_U=init_U,
                     init_b=init_b)

h3 = SimpleRecurrent(name='h3',
                     parent=['x', 'h2'],
                     batch_size=batch_size,
                     nout=200,
                     unit='tanh',
                     init_W=init_W,
                     init_U=init_U,
                     init_b=init_b)

h4 = FullyConnectedLayer(name='h4',
                         parent=['h1', 'h2', 'h3'],
                         nout=res,
                         unit='sigmoid',
                         init_W=init_W,
                         init_b=init_b)

cost = MSELayer(name='cost', parent=['h4', 'y'])

nodes = [h1, h2, h3, h4, cost]
rnn = Net(inputs=inputs, inputs_dim=inputs_dim, nodes=nodes)
cost = unpack(rnn.build_recurrent_graph(output_args=[cost]))
cost = cost.mean()
cost.name = 'cost'
model.graphs = [rnn]

optimizer = Adam(
    lr=0.01
)

extension = [
    GradientClipping(batch_size=batch_size),
    EpochCount(100),
    Monitoring(freq=100,
               ddout=[cost]),
    Picklize(freq=200,
             path=save_path)
]

mainloop = Training(
    name='toy_bb',
    data=Iterator(trdata, batch_size),
    model=model,
    optimizer=optimizer,
    cost=cost,
    outputs=[cost],
    extension=extension
)
mainloop.run()
