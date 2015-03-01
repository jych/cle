import ipdb
import numpy as np

from cle.cle.graph.net import Net
from cle.cle.layers import (
    InputLayer,
    OnehotLayer,
    MulCrossEntropyLayer,
    InitCell
)
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.layers.conv import ConvertLayer, Conv2DLayer
from cle.cle.train import Training
from cle.cle.train.ext import (
    EpochCount,
    GradientClipping,
    Monitoring,
    Picklize
)
from cle.cle.train.opt import Adam
from cle.cle.utils import error, predict
from cle.datasets.cifar10 import CIFAR10


# Toy example to use cle!

# Set your dataset
#datapath = '/data/lisa/data/mnist/mnist.pkl'
#savepath = '/u/chungjun/repos/cle/saved/'
datapath = ['/home/junyoung/data/cifar10/pylearn2_gcn_whitened/train.npy',
            '/home/junyoung/data/cifar10/pylearn2_gcn_whitened/trainy.npy']
savepath = '/home/junyoung/repos/cle/saved/'

batch_size = 100

trdata = CIFAR10(name='train',
                 path=datapath,
                 batch_size=batch_size)

# Choose the random initialization method
init_W, init_b = InitCell('randn'), InitCell('zeros')

# Define nodes: objects
inp, tar = trdata.theano_vars()
x = InputLayer(name='x', root=inp, nout=3072)
y = InputLayer(name='y', root=tar, nout=10)
c1 = ConvertLayer(name='c1',
                  parent=[x],
                  outshape=(batch_size, 3, 32, 32))
h1 = Conv2DLayer(name='h1',
                 parent=[c1],
                 outshape=(batch_size, 32, 16, 16),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)
h2 = Conv2DLayer(name='h2',
                 parent=[h1],
                 outshape=(batch_size, 32, 3, 3),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)
c2 = ConvertLayer(name='c2',
                  parent=[h2],
                  outshape=(batch_size, 100))
h3 = FullyConnectedLayer(name='h3',
                         parent=[c2],
                         nout=10,
                         unit='softmax',
                         init_W=init_W,
                         init_b=init_b)
cost = MulCrossEntropyLayer(name='cost', parent=[y, h3])

# You will fill in a list of nodes and fed them to the model constructor
nodes = [x, y, c1, h1, h2, c2, h3, cost]

# Your model will build the Theano computational graph
model = Net(nodes=nodes)
model.build_graph()

# You can access any output of a node by simply doing model.nodes[$node_name].out
cost = model.nodes['cost'].out
err = error(predict(model.nodes['h3'].out), predict(model.nodes['y'].out))
cost.name = 'cost'
err.name = 'error_rate'

# Define your optimizer: Momentum (Nesterov), RMSProp, Adam
optimizer = Adam(
    lr=0.001
)

extension = [
    GradientClipping(batch_size),
    EpochCount(100),
    Monitoring(freq=100,
               ddout=[cost, err]),
    Picklize(freq=10,
             path=savepath)
]

mainloop = Training(
    name='toy_cifar',
    data=trdata,
    model=model,
    optimizer=optimizer,
    cost=cost,
    outputs=[cost, err],
    extension=extension
)
mainloop.run()
