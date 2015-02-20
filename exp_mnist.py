import ipdb
from net import *
from util import *

try:
    datapath = '/data/lisa/data/mnist/mnist.pkl'
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = np.load(datapath)
except IOError:
    datapath = '/home/junyoung/data/mnist/mnist.pkl'
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = np.load(datapath)

batch_size = 128
num_batches = train_x.shape[0] / batch_size

init_W, init_b = ParamInit('randn'), ParamInit('zeros')

X = T.fmatrix()
y = one_hot(train_y)
l1 = FullyConnectedLayer(name='h1',
                         n_in=784,
                         n_out=1000,
                         unit='relu',
                         init_W=init_W,
                         init_b=init_b)

l2 = FullyConnectedLayer(name='h2',
                         n_in=1000,
                         n_out=10,
                         unit='softmax',
                         init_W=init_W,
                         init_b=init_b)

net = SeqNet('net', l1, l2)
cost = NLL_mul(net.fprop(X), y)

ipdb.set_trace()
train_fn = theano.function(
    inputs=[X, y],
    outputs=[cost],
    on_unused_input='ignore',
    updates=rms_prop({W1: g_W1, B1: g_B1, V1: g_V1, C1: g_C1}, __lr)
)

ipdb.set_trace()
for i in xrange(num_batches):
    indices = np.arange(i*batch_size, (i+1)*batch_size)
    cost += train_fn(train_x[indices, :], y)
