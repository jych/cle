import ipdb
import numpy as np
import theano

from cle.cle.cost import NllMulInd
from cle.cle.data import Iterator
from cle.cle.models import Model
from cle.cle.layers import InitCell
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.train import Training
from cle.cle.train.ext import (
    EpochCount,
    GradientClipping,
    Monitoring,
    Picklize,
    WeightNorm
)
from cle.cle.train.opt import RMSProp
from cle.cle.utils import error, init_tparams, flatten, predict, OrderedDict
from cle.cle.utils.op import dropout, add_noise_params
from cle.datasets.mnist import MNIST


# Regularization parameters
std_dev = 0.001
inp_p = 1.0
inp_scale = 1 / inp_p
int_p = 0.5
int_scale = 1 / int_p


# Set your dataset
data_path = '/data/lisa/data/mnist/mnist.pkl'
save_path = '/u/chungjun/src/cle/saved/'

batch_size = 128
debug = 0

model = Model()
train_data = MNIST(name='train',
                   path=data_path)

valid_data = MNIST(name='valid',
                    path=data_path)

# Define nodes: objects
x, y = train_data.theano_vars()

# You must use THEANO_FLAGS="compute_test_value=raise" python -m ipdb
if debug:
    x.tag.test_value = np.zeros((batch_size, 784), dtype=np.float32)
    y.tag.test_value = np.zeros((batch_size, 1), dtype=np.float32)

# Choose the random initialization method
init_W = InitCell('rand')
init_b = InitCell('zeros')

h1 = FullyConnectedLayer(name='h1',
                         parent=['x'],
                         parent_dim=[784],
                         nout=1000,
                         unit='relu',
                         init_W=init_W,
                         init_b=init_b)

output = FullyConnectedLayer(name='output',
                             parent=['h1'],
                             parent_dim=[1000],
                             nout=10,
                             unit='softmax',
                             init_W=init_W,
                             init_b=init_b)


# You will fill in a list of nodes
nodes = [h1, output]

# Initalize the nodes
params = OrderedDict()
for node in nodes:
    params.update(node.initialize())
params = init_tparams(params)
nparams = add_noise_params(params, std_dev=std_dev)

# Build the Theano computational graph
d_x = inp_scale * dropout(x, p=inp_p)
h1_out = h1.fprop([d_x], nparams)
d1_out = int_scale * dropout(h1_out, p=int_p)
y_hat = output.fprop([d1_out], nparams)

# Compute the cost
cost = NllMulInd(y, y_hat).mean()
err = error(predict(y_hat), y)
cost.name = 'cross_entropy'
err.name = 'error_rate'

# Seperate computational graph to compute monitoring values without
# considering the noising processes
m_h1_out = h1.fprop([x], params)
m_y_hat = output.fprop([m_h1_out], params)

m_cost = NllMulInd(y, m_y_hat).mean()
m_err = error(predict(m_y_hat), y)
m_cost.name = 'cross_entropy'
m_err.name = 'error_rate'

monitor_fn = theano.function([x, y], [m_cost, m_err])

model.inputs = [x, y]
model.params = params
model.nodes = nodes

# Define your optimizer: Momentum (Nesterov), RMSProp, Adam
optimizer = RMSProp(
    #lr=0.01
    lr=0.005
)

extension = [
    GradientClipping(batch_size=batch_size, check_nan=1),
    EpochCount(500),
    Monitoring(freq=1000,
               ddout=[m_cost, m_err],
               data=[Iterator(train_data, batch_size),
                     Iterator(valid_data, batch_size)],
               monitor_fn=monitor_fn),
    Picklize(freq=100000, path=save_path),
    WeightNorm(keys='W')
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
