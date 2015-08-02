import ipdb
import numpy as np
import theano

from cle.cle.cost import NllMulInd
from cle.cle.data import Iterator
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
from cle.cle.utils import error, flatten, predict, OrderedDict
from cle.datasets.mnist import MNIST

# Set your dataset
data_path = '/home/junyoung/data/mnist/mnist.pkl'
save_path = '/home/junyoung/repos/cle/saved/'

batch_size = 128
debug = 0

model = Model()
train_data = MNIST(name='train',
                   path=data_path)

valid_data = MNIST(name='valid',
                    path=data_path)

# Choose the random initialization method
init_W = InitCell('rand')
init_b = InitCell('zeros')

# Define nodes: objects
x, y = train_data.theano_vars()
mn_x, mn_y = valid_data.theano_vars()
# You must use THEANO_FLAGS="compute_test_value=raise" python -m ipdb
if debug:
    x.tag.test_value = np.zeros((batch_size, 784), dtype=np.float32)
    y.tag.test_value = np.zeros((batch_size, 1), dtype=np.float32)
    mn_x.tag.test_value = np.zeros((batch_size, 784), dtype=np.float32)
    mn_y.tag.test_value = np.zeros((batch_size, 1), dtype=np.float32)

h1 = FullyConnectedLayer(name='h1',
                         parent=['x'],
                         parent_dim=[784],
                         nout=1000,
                         unit='relu',
                         init_W=init_W,
                         init_b=init_b)

d1 = DropoutLayer(name='d1', parent=['h1'], nout=1000)

h2 = FullyConnectedLayer(name='h2',
                         parent=['d1'],
                         parent_dim=[1000],
                         nout=1000,
                         unit='relu',
                         init_W=init_W,
                         init_b=init_b)

d2 = DropoutLayer(name='d2', parent=['h2'], nout=1000)

output = FullyConnectedLayer(name='output',
                             parent=['d2'],
                             parent_dim=[1000],
                             nout=10,
                             unit='softmax',
                             init_W=init_W,
                             init_b=init_b)


# You will fill in a list of nodes
nodes = [h1, h2, d1, d2, output]

# Initalize the nodes
for node in nodes:
    node.initialize()

# Collect parameters
params = flatten([node.get_params().values() for node in nodes])

# Build the Theano computational graph
h1_out = h1.fprop([x])
d1_out = d1.fprop([h1_out])
h2_out = h2.fprop([d1_out])
d2_out = d2.fprop([h2_out])
y_hat = output.fprop([d2_out])

# Compute the cost
cost = NllMulInd(y, y_hat).mean()
err = error(predict(y_hat), y)
cost.name = 'cross_entropy'
err.name = 'error_rate'

d1.set_mode(1)
d2.set_mode(1)
mn_h1_out = h1.fprop([mn_x])
mn_d1_out = d1.fprop([mn_h1_out])
mn_h2_out = h2.fprop([mn_d1_out])
mn_d2_out = d2.fprop([mn_h2_out])
mn_y_hat = output.fprop([mn_d2_out])

mn_cost = NllMulInd(mn_y, mn_y_hat).mean()
mn_err = error(predict(mn_y_hat), mn_y)
mn_cost.name = 'cross_entropy'
mn_err.name = 'error_rate'

monitor_fn = theano.function([mn_x, mn_y], [mn_cost, mn_err])

model.inputs = [x, y]
model._params = params
model.nodes = nodes

# Define your optimizer: Momentum (Nesterov), RMSProp, Adam
optimizer = RMSProp(
    lr=0.001
)

extension = [
    GradientClipping(),
    EpochCount(500),
    Monitoring(freq=100,
               ddout=[mn_cost, mn_err],
               data=[Iterator(train_data, batch_size),
                     Iterator(valid_data, batch_size)],
               monitor_fn=monitor_fn),
    Picklize(freq=1000000,
             path=save_path),
    WeightNorm(param_name='W')
]

mainloop = Training(
    name='toy_mnist_dropout',
    data=Iterator(train_data, batch_size),
    model=model,
    optimizer=optimizer,
    cost=cost,
    outputs=[cost, err],
    extension=extension
)
mainloop.run()
