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
from cle.cle.layers.layer import MaxPool2D
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
#datapath = '/data/lisa/data/cifar10/pylearn2_gcn_whitend/train.npy'
#savepath = '/u/chungjun/repos/cle/saved/'
datapath = ['/home/junyoung/data/cifar10/pylearn2_gcn_whitened/train.npy',
            '/home/junyoung/data/cifar10/pylearn2_gcn_whitened/trainy.npy']
testdatapath = ['/home/junyoung/data/cifar10/pylearn2_gcn_whitened/test.npy',
                '/home/junyoung/data/cifar10/pylearn2_gcn_whitened/testy.npy']
savepath = '/home/junyoung/repos/cle/saved/'

batch_size = 100

trdata = CIFAR10(name='train',
                 path=datapath,
                 batch_size=batch_size)
testdata = CIFAR10(name='test',
                   path=testdatapath,
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
                 outshape=(batch_size, 96, 30, 30),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)
h2 = Conv2DLayer(name='h2',
                 parent=[h1],
                 outshape=(batch_size, 96, 28, 28),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)
p1 = MaxPool2D(name='p1',
               parent=[h2],
               poolsize=(3, 3),
               poolstride=(2, 2))
h3 = Conv2DLayer(name='h3',
                 parent=[p1],
                 outshape=(batch_size, 192, 11, 11),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)
h4 = Conv2DLayer(name='h4',
                 parent=[h3],
                 outshape=(batch_size, 192, 9, 9),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)
h5 = Conv2DLayer(name='h5',
                 parent=[h4],
                 outshape=(batch_size, 192, 7, 7),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)
p2 = MaxPool2D(name='p2',
               parent=[h5],
               poolsize=(3, 3),
               poolstride=(2, 2))
h6 = Conv2DLayer(name='h6',
                 parent=[p2],
                 outshape=(batch_size, 192, 2, 2),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)
c2 = ConvertLayer(name='c2',
                  parent=[h6],
                  outshape=(batch_size, 768))
h7 = FullyConnectedLayer(name='h7',
                         parent=[c2],
                         nout=10,
                         unit='softmax',
                         init_W=init_W,
                         init_b=init_b)
cost = MulCrossEntropyLayer(name='cost', parent=[y, h7])

# You will fill in a list of nodes and fed them to the model constructor
nodes = [x, y, c1, c2, h1, h2, h3, h4, h5, h6, h7, p1, p2, cost]

# Your model will build the Theano computational graph
model = Net(nodes=nodes)
model.build_graph()

# You can access any output of a node by simply doing model.nodes[$node_name].out
cost = model.nodes['cost'].out
err = error(predict(model.nodes['h7'].out), predict(model.nodes['y'].out))
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
               ddout=[cost, err],
               data=[testdata]),
    #Picklize(freq=10,
    #         path=savepath)
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
