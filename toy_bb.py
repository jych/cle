import ipdb
import theano
import theano.tensor as T
import time

from cost import *
from data import *
from ext import *
from layer import *
from opt import *
from net import *
from train import *
from util import *
from mnist import MNIST


# Toy example to use cle!

# Set your dataset
try:
    datapath = '/data/lisatmp3/chungjun/bouncing_balls/bouncing_ball_2balls_16wh_20len_50000cases.npy'
    (tr_x, tr_y), (val_x, val_y), (test_x, test_y) = np.load(datapath)
except IOError:
    datapath = '/home/junyoung/data/bouncing_balls/bouncing_ball_2balls_16wh_20len_50000cases.npy'
    tr_x = np.load(datapath)
savepath = '/home/junyoung/repos/cle/saved/'

batch_size = 128
num_batches = tr_x.shape[0] / batch_size

trdata = BouncingBalls(name='train',
                       data=tr_x,
                       batch_size=batch_size)

# Choose the random initialization method
init_W, init_U, init_b = ParamInit('randn'), ParamInit('ortho'), ParamInit('zeros')

# Define nodes: objects
inp, tar = trdata.theano_vars()
x = Input(inp)
y = Input(tar)
proj = IdentityLayer()
onehot = IdentityLayer()
h1 = RecurrentLayer(name='h1',
                    n_in=256,
                    n_out=200,
                    unit='tanh',
                    init_W=init_W,
                    init_U=init_U,
                    init_b=init_b)

h2 = RecurrentLayer(name='h2',
                    n_in=256,
                    n_out=200,
                    unit='tanh',
                    init_W=init_W,
                    init_U=init_U,
                    init_b=init_b)
cost = MSELayer()

# You will fill in your node and edge lists
# and fed them to the model constructor
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
# based on topological sorting on given nodes and edges
# It will Build a DAG using depth-first search
# Your model is smart enough to take care of the rest
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
cost.name = 'cost'
err.name = 'error_rate'

# Define your optimizer [Momentum(Nesterov) / RMSProp / Adam]
optimizer = RMSProp(
#optimizer = Adam(
    lr=0.001
)

extension = [
    GradientClipping(),
    EpochCount(40),
    Monitoring(freq=100,
               ddout=[cost, err],
               data=[valdata]),
    Picklize(freq=10,
             path=savepath)
]

toy_mnist = Training(
    name='toy_mnist',
    data=trdata,
    model=model,
    optimizer=optimizer,
    cost=cost,
    outputs=[cost, err],
    extension=extension
)
toy_mnist.run()
