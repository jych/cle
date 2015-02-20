import ipdb
from net import *
from util import *

datapath = '/data/lisa/data/mnist/mnist.pkl'
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = np.load(datapath)

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
    inputs=[X],
    outputs=[cost],
    on_unused_input='ignore',
    updates=rms_prop({W1: g_W1, B1: g_B1, V1: g_V1, C1: g_C1}, __lr)
)
ipdb.set_trace()
