import ipdb
import numpy as np

from cle.cle.graph.net import Net
from cle.cle.data import Iterator
from cle.cle.models import Model
from cle.cle.layers import InitCell
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
from cle.cle.utils import error, flatten, predict
from cle.datasets.cifar10 import CIFAR10

# Set your dataset
data_path = ['/data/lisa/data/cifar10/pylearn2_gcn_whitened/train.npy',
            '/u/chungjun/repos/cle/labels/trainy.npy']
test_data_path = ['/data/lisa/data/cifar10/pylearn2_gcn_whitened/test.npy',
                '/u/chungjun/repos/cle/labels/testy.npy']
save_path = '/u/chungjun/repos/cle/saved/'
#data_path = ['/home/junyoung/data/cifar10/pylearn2_gcn_whitened/train.npy',
#            '/home/junyoung/data/cifar10/pylearn2_gcn_whitened/trainy.npy']
#test_data_path = ['/home/junyoung/data/cifar10/pylearn2_gcn_whitened/test.npy',
#                '/home/junyoung/data/cifar10/pylearn2_gcn_whitened/testy.npy']
#save_path = '/home/junyoung/repos/cle/saved/'

batch_size = 128
debug = 0

model = Model()
train_data = CIFAR10(name='train',
                     path=data_path)

test_data = CIFAR10(name='test',
                    path=test_data_path)

# Choose the random initialization method
init_W = InitCell('rand')
init_b = InitCell('zeros')

# Define nodes: objects
model.inputs = train_data.theano_vars()
x, y = model.inputs
# You must use THEANO_FLAGS="compute_test_value=raise" python -m ipdb
if debug:
    x.tag.test_value = np.zeros((batch_size, 3072), dtype=np.float32)
    y.tag.test_value = np.zeros((batch_size, 10), dtype=np.float32)

inputs = [x, y]
inputs_dim = {'x':3072, 'y':10}

c1 = ConvertLayer(name='c1',
                  parent=['x'],
                  outshape=(batch_size, 3, 32, 32))

h1 = Conv2DLayer(name='h1',
                 parent=['c1'],
                 outshape=(batch_size, 64, 21, 21),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)

h2 = Conv2DLayer(name='h2',
                 parent=['h1'],
                 outshape=(batch_size, 128, 10, 10),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)

h3 = Conv2DLayer(name='h3',
                 parent=['h2'],
                 outshape=(batch_size, 128, 1, 1),
                 unit='relu',
                 init_W=init_W,
                 init_b=init_b)

c2 = ConvertLayer(name='c2',
                  parent=['h3'],
                  outshape=(batch_size, 128))

# Global average pooling missing
h4 = FullyConnectedLayer(name='h4',
                         parent=['c2'],
                         nout=10,
                         unit='softmax',
                         init_W=init_W,
                         init_b=init_b)

cost = MulCrossEntropyLayer(name='cost', parent=['y', 'h4'])

# You will fill in a list of nodes and fed them to the model constructor
nodes = [c1, c2, h1, h2, h3, h4, cost]

# Your model will build the Theano computational graph
cnn = Net(inputs=inputs, inputs_dim=inputs_dim, nodes=nodes)
cnn.build_graph()

# You can access any output of a node by doing model.nodes[$node_name].out
cost = cnn.nodes['cost'].out
err = error(predict(cnn.nodes['h4'].out), predict(y))
cost.name = 'cost'
err.name = 'error_rate'
model.graphs = [cnn]

# Define your optimizer: Momentum (Nesterov), RMSProp, Adam
optimizer = Adam(
    #lr=0.00005
    lr=0.0005
)

extension = [
    GradientClipping(batch_size=batch_size),
    EpochCount(100),
    Monitoring(freq=100,
               ddout=[cost, err],
               data=[Iterator(test_data, batch_size)]),
    Picklize(freq=10000,
             path=save_path)
]

mainloop = Training(
    name='toy_cifar',
    data=Iterator(train_data, batch_size),
    model=model,
    optimizer=optimizer,
    cost=cost,
    outputs=[cost, err],
    extension=extension
)
mainloop.run()
