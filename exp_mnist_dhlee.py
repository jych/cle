import ipdb
import theano
import theano.tensor as T

from net import *
from util import *
from cost import *
from layer import *
from data import *
from opt import *

try:
    datapath = '/data/lisa/data/mnist/mnist.pkl'
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = np.load(datapath)
except IOError:
    #datapath = '/home/junyoung/data/mnist/mnist.pkl'
    datapath = '/work/mnist/mnist.pkl'
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = np.load(datapath)

# data

batch_iter = BatchProvider( data_list = ( DesignMatrix(train_x), DesignMatrix(one_hot(train_y)) ),
                            batch_size = 100 )

init_W, init_b = ParamInit('randn',0,0.01), ParamInit('zeros')

X, Y = T.fmatrices(2)
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

net = SeqNet('net', h1, h2)
P = net.fprop(X)

cost = NLLMul(P, Y)
err = error( predict(P), predict(Y) )

grad_params = T.grad( cost, net.params )

optimizer = RMSprop(0.001)

ipdb.set_trace()
train_fn = theano.function(
    inputs=[X, Y],
    outputs=[cost, err],
    on_unused_input='ignore',
    updates=optimizer.updates(cost, net.params)
)

ipdb.set_trace()
for e in xrange(100) :
    for data_batch in batch_iter :
        train_fn(*data_batch)




