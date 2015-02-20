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
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = np.load(datapath)
except IOError:
    datapath = '/home/junyoung/data/mnist/mnist.pkl'
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = np.load(datapath)

batch_size = 128
num_batches = train_x.shape[0] / batch_size
batch_iter = BatchProvider(data_list=(DesignMatrix(train_x), DesignMatrix(one_hot(train_y))),
                           batch_size=batch_size)

init_W, init_b = ParamInit('randn'), ParamInit('zeros')
inputs = T.fmatrix()
targets = T.fmatrix()

x = IdentityLayer()
y = OnehotLayer(max_labels=10)
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

# Build DAG based on depth-first search
# reference: B & B page 301
nodes = {'x':x, 'h1':h1, 'h2':h2, 'y':y, 'cost':cost}
edges = {'h1':'x', 'h2':'h1', 'cost':['h2', 'y']}
model = Net(nodes=nodes, edges=edges)
cost = model.compute_cost(inputs, targets)

ipdb.set_trace()
cost_fn = theano.function(
    inputs=[x, y],
    outputs=[cost],
    on_unused_input='ignore',
    updates=rms_prop({W1: g_W1, B1: g_B1, V1: g_V1, C1: g_C1}, __lr)
)

for data_batch in batch_iter:
    cost += cost_fn(*data_batch)
cost /= num_batches
print cost
