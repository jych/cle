import ipdb
import numpy as np

from cle.cle.data import Iterator
from cle.cle.graph.net import Net
from cle.cle.layers import (
    InputLayer,
    OnehotLayer,
    InitCell
)
from cle.cle.layers.cost import MulCrossEntropyLayer
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


# Set your dataset
#datapath = '/data/lisa/data/cifar10/pylearn2_gcn_whitend/train.npy'
#savepath = '/u/chungjun/repos/cle/saved/'
datapath = ['/home/junyoung/data/cifar10/pylearn2_gcn_whitened/train.npy',
            '/home/junyoung/data/cifar10/pylearn2_gcn_whitened/trainy.npy']
testdatapath = ['/home/junyoung/data/cifar10/pylearn2_gcn_whitened/test.npy',
                '/home/junyoung/data/cifar10/pylearn2_gcn_whitened/testy.npy']
savepath = '/home/junyoung/repos/cle/saved/'

batchsize = 128
debug = 0

trdata = CIFAR10(name='train',
                 path=datapath)
testdata = CIFAR10(name='test',
                   path=testdatapath)

# Choose the random initialization method
init_W = InitCell('randn')
init_b = InitCell('zeros')

# Define nodes: objects
inp, tar = trdata.theano_vars()
# You must use THEANO_FLAGS="compute_test_value=raise" python -m ipdb
if debug:
    inp.tag.test_value = np.zeros((batchsize, 3072), dtype=np.float32)
    tar.tag.test_value = np.zeros((batchsize, 10), dtype=np.float32)
x = InputLayer(name='x', root=inp, nout=3072)
y = InputLayer(name='y', root=tar, nout=10)
c1 = ConvertLayer(name='c1',
                  parent=[x],
                  outshape=(batchsize, 3, 32, 32))
h1 = Conv2DLayer(name='h1',
                 parent=[c1],
                 outshape=(batchsize, 96, 30, 30),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)
h2 = Conv2DLayer(name='h2',
                 parent=[h1],
                 outshape=(batchsize, 96, 28, 28),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)
p1 = MaxPool2D(name='p1',
               parent=[h2],
               poolsize=(3, 3),
               poolstride=(2, 2))
h3 = Conv2DLayer(name='h3',
                 parent=[p1],
                 outshape=(batchsize, 192, 11, 11),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)
h4 = Conv2DLayer(name='h4',
                 parent=[h3],
                 outshape=(batchsize, 192, 9, 9),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)
h5 = Conv2DLayer(name='h5',
                 parent=[h4],
                 outshape=(batchsize, 192, 7, 7),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)
p2 = MaxPool2D(name='p2',
               parent=[h5],
               poolsize=(3, 3),
               poolstride=(2, 2))
h6 = Conv2DLayer(name='h6',
                 parent=[p2],
                 outshape=(batchsize, 192, 1, 1),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)
c2 = ConvertLayer(name='c2',
                  parent=[h6],
                  outshape=(batchsize, 192))
# Global average pooling missing
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

# You can access any output of a node by doing model.nodes[$node_name].out
cost = model.nodes['cost'].out
err = error(predict(model.nodes['h7'].out), predict(model.nodes['y'].out))
cost.name = 'cost'
err.name = 'error_rate'

# Define your optimizer: Momentum (Nesterov), RMSProp, Adam
optimizer = Adam(
    lr=0.1
)

extension = [
    GradientClipping(batchsize=batchsize),
    EpochCount(100),
    Monitoring(freq=100,
               ddout=[cost, err],
               data=[Iterator(testdata, batchsize)]),
    Picklize(freq=10,
             path=savepath)
]

mainloop = Training(
    name='toy_cifar',
    data=Iterator(trdata, batchsize),
    model=model,
    optimizer=optimizer,
    cost=cost,
    outputs=[cost, err],
    extension=extension
)
mainloop.run()
