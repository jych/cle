import ipdb
import theano
import theano.tensor as T
import time

from cost import *
from data import *
from ext import *
from layer import *
from monitor import Monitor
from opt import *
from net import *
from train import *
from util import *
from mnist import MNIST


# Toy example to use cle!

# Set your dataset
try:
    datapath = '/data/lisa/data/mnist/mnist.pkl'
    (tr_x, tr_y), (val_x, val_y), (test_x, test_y) = np.load(datapath)
except IOError:
    datapath = '/home/junyoung/data/mnist/mnist.pkl'
    (tr_x, tr_y), (val_x, val_y), (test_x, test_y) = np.load(datapath)

batch_size = 128
num_batches = tr_x.shape[0] / batch_size
trdata = MNIST(name='train',
               data=(tr_x, tr_y),
               batch_size=batch_size)
valdata = MNIST(name='valid',
                data=(val_x, val_y),
                batch_size=batch_size)

# Choose the random initialization method
init_W, init_b = ParamInit('randn'), ParamInit('zeros')

# Define nodes: objects
inp, tar = trdata.theano_vars()
x = Input(inp)
y = Input(tar)
proj = IdentityLayer()
onehot = OnehotLayer(max_labels=10)
h1 = FullyConnectedLayer(n_in=784,
                         n_out=1000,
                         unit='relu',
                         init_W=init_W,
                         init_b=init_b)

h2 = FullyConnectedLayer(n_in=1000,
                         n_out=10,
                         unit='softmax',
                         init_W=init_W,
                         init_b=init_b)
cost = MulCrossEntropyLayer()

# Topological sorting on directed acyclic graph (DAG)
# Build DAG based on depth-first search
# You will fill in your node and edge lists
# and fed them to the model constructor
# Your model is smart enough to take care of the rest
nodes = {
    'x': x,
    #'y': y,
    'proj': proj,
    #'onehot': onehot,
    'h1': h1,
    'h2': h2,
    #'cost': cost
}
edges = {
    'x': 'proj',
    #'y': 'onehot',
    'proj': 'h1',
    'h1': 'h2',
    #'h2': 'cost',
    #'onehot': 'cost'
}

# Your model will build the Theano computational graph
# Based on given nodes and edges
model = Net(nodes=nodes, edges=edges)

# You have already defined your nodes and edges
# But you want to add another nodes and edges
# It's not too late, add the nodes and edges
# Then, build the Theano computational graph again
model.add_node({'y': y, 'onehot': onehot, 'cost': cost})
model.add_edge({'y': 'onehot', 'onehot': 'cost', 'h2': 'cost'})
model.build_graph()

# You can access any output of a node by simply doing
# model.nodes[$node_name].out
# This is the most cool part :)
# Super easy to monitor cost and states with same function
cost = model.nodes['cost'].out
err = error(predict(model.nodes['h2'].out), predict(model.nodes['onehot'].out))

# Define your optimizer [Momentum(Nesterov) / RMSProp / Adam]
optimizer = RMSProp(
    learning_rate=0.001
)

extension = [
    GradientClipping(),
    EpochCount(40)
]
monitor = Monitor(
    model=model,
    freq=100,
    #opt_ch=['cost', 'err'],
    dd_ch=['cost', 'err'],
    data=[trdata, valdata]
)
monitor.build_monitor_graph(outputs=[cost, err])

toy_mnist = Training(
    data=trdata,
    model=model,
    optimizer=optimizer,
    outputs=[cost, err],
    monitor=monitor,
    extension=extension
)
toy_mnist.run()

# What are not done yet
# 1. Monitoring
# 2. Serialization / Checkpoint
# 3. Dropout / Regularization: we should implement dropout using Theano.clone
# 4. RNN
# 5. CNN
# 6. VAE
# 7. Predefined nets: larger building block such as MLP, ConvNet and Stacked RNN
