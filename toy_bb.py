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
from bouncing_balls import BouncingBalls


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
x = Input('h1_inp1', inp)
h1_0 = Input('h1_inp2', T.fmatrix())
h1 = RecurrentLayer('h2_inp1', 256, 200, 'tanh', init_W, init_U, init_b)
h2_0 = Input('h2_inp2', T.fmatrix())
h2 = RecurrentLayer('cost_inp1', 200, 200, 'tanh', init_W, init_U, init_b)
y = Input('cost_inp2', tar)
cost = MSELayer('cost')
nodes = {
    'x': x,
    'y': y,
    'h1': h1,
    'h2': h2,
    'h1_0': h1_0,
    'h2_0': h2_0,
    'cost': cost
}
edges = {
    'x': 'h1',
    'h1_0': 'h1', 
    'h1': 'h2',
    'h2_0': 'h2',
    'h2': 'cost',
    'y': 'cost'
}
model = Net(nodes=nodes, edges=edges)
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
