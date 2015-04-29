import ipdb
import numpy as np
import theano

from cle.cle.data import Iterator
from cle.cle.graph.net import Net
from cle.cle.models import Model
from cle.cle.layers import InitCell, OnehotLayer
from cle.cle.layers.cost import MulCrossEntropyLayer
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.layers.layer import DropoutLayer
from cle.cle.train import Training
from cle.cle.train.ext import (
    EpochCount,
    GradientClipping,
    Monitoring,
    Picklize,
    WeightNorm
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
init_W = InitCell('rand')
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

d1 = DropoutLayer(name='d1', parent=['h1'], nout=1000)

h2 = FullyConnectedLayer(name='h2',
                         parent=['d1'],
                         nout=1000,
                         unit='relu',
                         init_W=init_W,
                         init_b=init_b)

d2 = DropoutLayer(name='d2', parent=['h2'], nout=1000)

h3 = FullyConnectedLayer(name='h3',
                         parent=['d2'],
                         nout=10,
                         unit='softmax',
                         init_W=init_W,
                         init_b=init_b)

cost = MulCrossEntropyLayer(name='cost', parent=['onehot', 'h3'])

# You will fill in a list of nodes and fed them to the model constructor
nodes = [onehot, h1, h2, h3, d1, d2, cost]

# Your model will build the Theano computational graph
mlp = Net(inputs=inputs, inputs_dim=inputs_dim, nodes=nodes)
mlp.build_graph()

# You can access any output of a node by doing model.nodes[$node_name].out
cost_ = mlp.nodes['cost'].out
err = error(predict(mlp.nodes['h3'].out), predict(mlp.nodes['onehot'].out))
cost_.name = 'cost'
err.name = 'error_rate'
model.graphs = [mlp]

mlp.nodes['d1'].set_mode(1)
mlp.nodes['d2'].set_mode(1)
mlp.build_graph()
mn_cost = mlp.nodes['cost'].out
mn_cost.name = 'mn_cost'
mn_err = error(predict(mlp.nodes['h3'].out), predict(mlp.nodes['onehot'].out))
mn_err.name = 'mn_err'
monitor_fn = theano.function(inputs, [mn_cost, mn_err])

# Define your optimizer: Momentum (Nesterov), RMSProp, Adam
optimizer = RMSProp(
    lr=0.001
)

extension = [
    GradientClipping(),
    EpochCount(500),
    Monitoring(freq=100,
               ddout=[mn_cost, mn_err],
               data=[Iterator(trdata, batch_size),
                     Iterator(valdata, batch_size)],
               monitor_fn=monitor_fn),
    Picklize(freq=1000000,
             path=save_path),
    WeightNorm(param_name='W')    
]

mainloop = Training(
    name='toy_mnist_dropout',
    data=Iterator(trdata, batch_size),
    model=model,
    optimizer=optimizer,
    cost=cost_,
    outputs=[cost_, err],
    extension=extension
)
mainloop.run()
