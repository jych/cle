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
    datapath = '/data/lisa/data/mnist/mnist.pkl'
    (tr_x, tr_y), (val_x, val_y), (test_x, test_y) = np.load(datapath)
except IOError:
    datapath = '/home/junyoung/data/mnist/mnist.pkl'
    (tr_x, tr_y), (val_x, val_y), (test_x, test_y) = np.load(datapath)
savepath = '/home/junyoung/repos/cle/saved/'

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
x = Input(name='x', inp=inp)
y = Input(name='y', inp=tar)
onehot = OnehotLayer('onehot', max_labels=10)
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

# You will fill in your node and edge lists
# and fed them to the model constructor
graph = [
    [x, h1],
    [h1, h2],
    [[onehot, h2], cost],
    [y, onehot]
]

# Your model will build the Theano computational graph
# based on topological sorting on given nodes and edges
# It will Build a DAG using depth-first search
# Your model is smart enough to take care of the rest
model = Net(graph=graph)
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

# What are not done yet
# 1. Monitoring                      done!
# 2. Serialization / Checkpoint      done! Thanks to kastnerkyle and Blocks
#                                    working on early stopping
# 3. Dropout: use Theano.clone
# 4. Other Regularization
# 5. RNN                             jych is doing
# 6. CNN                             donghyunlee is doing
# 7. VAE                             laurent-dinh????????? :)
# 8. Predefined nets: larger building block such as MLP, ConvNet and Stacked RNN
