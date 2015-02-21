import ipdb
import theano
import theano.tensor as T

from net import *
from util import *
from cost import *
from layer import *
from opt import *
from data import *

try:
    datapath = '/data/lisa/data/mnist/mnist.pkl'
    (tr_x, tr_y), (val_x, val_y), (test_x, test_y) = np.load(datapath)
except IOError:
    datapath = '/home/junyoung/data/mnist/mnist.pkl'
    (tr_x, tr_y), (val_x, val_y), (test_x, test_y) = np.load(datapath)

batch_size = 128
num_batches = tr_x.shape[0] / batch_size
batch_iter = BatchProvider(data_list=(DesignMatrix(tr_x),
                                      DesignMatrix(one_hot(tr_y))),
                           batch_size=batch_size)

init_W, init_b = ParamInit('randn'), ParamInit('zeros')
inp = T.fmatrix()
#tar = T.fmatrix()
tar = T.lvector()
x = Input(inp)
y = Input(tar)

proj = IdentityLayer()
onehot = OnehotLayer(max_labels=10)
h1 = FullyConnectedLayer(name='h1',
                         n_in=784,
                         n_out=1000,
                         unit='relu',
                         init_W=init_W,
                         init_b=init_b)

h2 = FullyConnectedLayer(name='h2',
                         n_in=1000,
                         n_out=10,
                         unit='softmax',
                         init_W=init_W,
                         init_b=init_b)
cost = MulCrossEntropyLayer(name='cost')

# Topological sorting on directed acyclic graph (DAG)
# Build DAG based on depth-first search
nodes = {'proj': proj, 'onehot': onehot, 'x': x, 'h1': h1, 'h2': h2, 'y': y, 'cost': cost}
edges = {'x': 'proj', 'y': 'onehot', 'proj': 'h1', 'h1': 'h2', 'h2': 'cost', 'onehot': 'cost'}
model = Net(nodes=nodes, edges=edges)
cost = model.nodes['cost'].out
# You can access any output of a node by simply doing
# model.nodes[$node_name].out

optimizer = RMSProp(0.001)
cost_fn = theano.function(
    inputs=[inp, tar],
    outputs=[cost],
    on_unused_input='ignore',
    updates=optimizer.updates(cost, model.params)
)

for data_batch in batch_iter:
    cost += cost_fn(*data_batch)
cost /= num_batches
print cost
