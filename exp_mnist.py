import ipdb
import theano
import theano.tensor as T

from net import *
from util import *
from cost import *
from layer import *
from opt import *
from data import *


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
trbatch_iter = BatchProvider(data_list=(DesignMatrix(tr_x),
                                        DesignMatrix(tr_y)),
                             batch_size=batch_size)
valbatch_iter = BatchProvider(data_list=(DesignMatrix(val_x),
                                         DesignMatrix(val_y)),
                              batch_size=batch_size)


# Choose the random initialization method
init_W, init_b = ParamInit('randn'), ParamInit('zeros')

# Define nodes: objects
inp = T.fmatrix()
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
# You will fill in your node and edge lists
# and fed them to the model constructor
# Your model is smart enough to take care of the rest
nodes = {
    'x': x,
    'y': y,
    'proj': proj,
    'onehot': onehot,
    'h1': h1,
    'h2': h2,
    'cost': cost
}
edges = {
    'x': 'proj',
    'y': 'onehot',
    'proj': 'h1',
    'h1': 'h2',
    'h2': 'cost',
    'onehot': 'cost'
}
model = Net(nodes=nodes, edges=edges)

# You can access any output of a node by simply doing
# model.nodes[$node_name].out
# This is the most cool part :)
cost = model.nodes['cost'].out
err = error(predict(model.nodes['h2'].out), predict(model.nodes['onehot'].out))

# Define your optimizer [Momentum(Nesterov) / RMSProp / Adam]
#optimizer = Momentum(
#optimizer = RMSProp(
optimizer = Adam(
    learning_rate=0.001,
    gradient_clipping=1
)

# Compile your cost function
cost_fn = theano.function(
    inputs=[inp, tar],
    outputs=[cost, err],
    updates=optimizer.get_updates(cost, model.params),
    on_unused_input='ignore',
    allow_input_downcast=True
)

# Train loop
for e in xrange(40):
    tr_cost = 0
    tr_err = 0
    val_cost = 0
    val_err = 0
    for batch in trbatch_iter:
        this_cost, this_err = cost_fn(*batch)
        tr_cost += this_cost
        tr_err += this_err
    for batch in valbatch_iter:
        this_cost, this_err = cost_fn(*batch)
        val_cost += this_cost
        val_err += this_err
    tr_cost /= num_batches
    tr_err /= num_batches
    val_cost /= num_batches
    val_err /= num_batches
    print 'epoch: %d, tr_nll: %f, tr_err: %f, val_nll: %f, val_err: %f'\
        %(e + 1, tr_cost, tr_err, val_cost, val_err)


# Whar are need to be done
# 1. Monitoring
# 2. Serialization / Checkpoint
# 3. RNN
# 4. CNN
