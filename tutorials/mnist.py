import ipdb
import numpy as np

from cle.cle.data import Iterator
from cle.cle.graph.net import Net
from cle.cle.models import Model
from cle.cle.layers import InitCell, OnehotLayer
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
from cle.cle.utils import error, predict, OrderedDict
from cle.datasets.mnist import MNIST


# Set your dataset
#data_path = '/data/lisa/data/mnist/mnist.pkl'
#save_path = '/u/chungjun/repos/cle/saved/'
data_path = '/home/junyoung/data/mnist/mnist.pkl'
save_path = '/home/junyoung/repos/cle/saved/'

batch_size = 128
debug = 0

model = Model()
trdata = MNIST(name='train',
               path=data_path)
valdata = MNIST(name='valid',
                path=data_path)

# Choose the random initialization method
init_W = InitCell('randn')
init_b = InitCell('zeros')

# Define nodes: objects
model.inputs = trdata.theano_vars()
x, y = model.inputs
# You must use THEANO_FLAGS="compute_test_value=raise" python -m ipdb
if debug:
    x.tag.test_value = np.zeros((batch_size, 784), dtype=np.float32)
    y.tag.test_value = np.zeros((batch_size, 1), dtype=np.float32)

inputs = [x, y]
inputs_dim = {'x':784, 'y':1}

onehot = OnehotLayer(name='onehot',
                     parent=['y'],
                     nout=10)

h1 = FullyConnectedLayer(name='h1',
                         parent=['x'],
                         nout=1000,
                         unit='relu',
                         init_W=init_W,
                         init_b=init_b)

h2 = FullyConnectedLayer(name='h2',
                         parent=['h1'],
                         nout=10,
                         unit='softmax',
                         init_W=init_W,
                         init_b=init_b)

cost = MulCrossEntropyLayer(name='cost', parent=['onehot', 'h2'])

# You will fill in a list of nodes and fed them to the model constructor
nodes = [onehot, h1, h2, cost]

# Your model will build the Theano computational graph
mlp = Net(inputs=inputs, inputs_dim=inputs_dim, nodes=nodes)
mlp.build_graph()

# You can access any output of a node by doing model.nodes[$node_name].out
cost = mlp.nodes['cost'].out
err = error(predict(mlp.nodes['h2'].out), predict(mlp.nodes['onehot'].out))
cost.name = 'cost'
err.name = 'error_rate'
model.graphs = [mlp]

# Define your optimizer: Momentum (Nesterov), RMSProp, Adam
optimizer = RMSProp(
    lr=0.001
)

extension = [
    GradientClipping(),
    EpochCount(40),
    Monitoring(freq=100,
               ddout=[cost, err],
               data=[Iterator(trdata, batch_size),
                     Iterator(valdata, batch_size)]),
    Picklize(freq=200,
             path=save_path)
]

mainloop = Training(
    name='toy_mnist',
    data=Iterator(trdata, batch_size),
    model=model,
    optimizer=optimizer,
    cost=cost,
    outputs=[cost, err],
    extension=extension
)
mainloop.run()
