import ipdb
import numpy as np

from cle.cle.graph.net import Net
from cle.cle.layers import (
    InputLayer,
    OnehotLayer,
    InitCell
)
from cle.cle.layers.cost import MulCrossEntropyLayer
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.train import Training
from cle.cle.train.ext import (
    EpochCount,
    GradientClipping,
    Monitoring,
    Picklize,
    EarlyStopping
)
from cle.cle.train.opt import RMSProp
from cle.cle.utils import error, predict
from cle.datasets.mnist import MNIST


# Set your dataset
#datapath = '/data/lisa/data/mnist/mnist.pkl'
#savepath = '/u/chungjun/repos/cle/saved/'
datapath = '/home/junyoung/data/mnist/mnist.pkl'
savepath = '/home/junyoung/repos/cle/saved/'

batchsize = 128
debug = 0

trdata = MNIST(name='train',
               path=datapath,
               batchsize=batchsize)
valdata = MNIST(name='valid',
                path=datapath,
                batchsize=batchsize)

# Choose the random initialization method
init_W = InitCell('randn')
init_b = InitCell('zeros')

# Define nodes: objects
inp, tar = trdata.theano_vars()
# You must use THEANO_FLAGS="compute_test_value=raise" python -m ipdb
if debug:
    inp.tag.test_value = np.zeros((batchsize, 784), dtype=np.float32)
    tar.tag.test_value = np.zeros((batchsize, 1), dtype=np.float32)
x = InputLayer(name='x', root=inp, nout=784)
y = InputLayer(name='y', root=tar, nout=1)
onehot = OnehotLayer(name='onehot',
                     parent=[y],
                     nout=10)
h1 = FullyConnectedLayer(name='h1',
                         parent=[x],
                         nout=1000,
                         unit='relu',
                         init_W=init_W,
                         init_b=init_b)
h2 = FullyConnectedLayer(name='h2',
                         parent=[h1],
                         nout=10,
                         unit='softmax',
                         init_W=init_W,
                         init_b=init_b)
cost = MulCrossEntropyLayer(name='cost', parent=[onehot, h2])

# You will fill in a list of nodes and fed them to the model constructor
nodes = [x, y, onehot, h1, h2, cost]

# Your model will build the Theano computational graph
model = Net(nodes=nodes)
model.build_graph()

# You can access any output of a node by simply doing model.nodes[$node_name].out
cost = model.nodes['cost'].out
err = error(predict(model.nodes['h2'].out), predict(model.nodes['onehot'].out))
cost.name = 'cost'
err.name = 'error_rate'

# Define your optimizer: Momentum (Nesterov), RMSProp, Adam
optimizer = RMSProp(
    lr=0.1
)

extension = [
    GradientClipping(batchsize=batchsize),
    EpochCount(40),
    Monitoring(freq=100,
               ddout=[cost, err],
               data=[valdata]),
    Picklize(freq=200,
             path=savepath),
    EarlyStopping(path=savepath)
]

mainloop = Training(
    name='toy_mnist',
    data=trdata,
    model=model,
    optimizer=optimizer,
    cost=cost,
    outputs=[cost, err],
    extension=extension
)
mainloop.run()
