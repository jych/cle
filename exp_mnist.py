import net import *
import util import *

(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = np.load('/home/dhlee/ML/dataset/mnist/mnist.pkl')

init_W, init_B = ParamInit('randn',0,0.01), ParamInit('zeros')

X = DesignMatrixDataLayer('X', train_x, 100 )
Y = DesignMatrixDataLayer('Y', one_hot(train_y), 100 )
NN1 = NNLayer('hidden1', 784,  1000, 'relu',    init_W, init_B )
NN2 = NNLayer('hidden2', 1000, 10,   'softmax', init_W, init_B )

net = SeqNet('net', X, NN1, NN2) 

cost = NLL_mul( net.fprop(), Y.fprop() )


i = T.lscalar()
train_fn = theano.function( [i], [cost], 
	on_unused_input='ignore', 
	givens = givens_train(i), 
    updates=rms_prop( { W1 : g_W1, B1 : g_B1, V1 : g_V1, C1 : g_C1 }, __lr ) )