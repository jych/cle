import ipdb
import numpy as np
import theano
import theano.tensor as T

from cle.cle.data import Iterator
from cle.cle.cost import NllBin
from cle.cle.graph.net import Net
from cle.cle.models import Model
from cle.cle.models.draw import (
    CanvasLayer,
    ErrorLayer,
    ReadLayer,
    WriteLayer
)
from cle.cle.layers import InitCell, RealVectorLayer
from cle.cle.layers.conv import ConvertLayer
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.layers.layer import PriorLayer
from cle.cle.layers.recurrent import LSTM
from cle.cle.train import Training
from cle.cle.train.ext import (
    EpochCount,
    GradientClipping,
    Monitoring,
    Picklize,
    EarlyStopping
)
from cle.cle.train.opt import Adam
from cle.cle.utils import flatten
from cle.cle.utils.compat import OrderedDict
from cle.datasets.mnist import MNIST


datapath = '/home/junyoung/data/mnist/mnist.pkl'
savepath = '/home/junyoung/repos/cle/saved/'

batch_size = 128
inpsz = 784
latsz = 10
n_steps = 10
debug = 1

model = Model()
data = MNIST(name='train',
             unsupervised=1,
             path=datapath)

init_W = InitCell('rand')
init_U = InitCell('ortho')
init_b = InitCell('zeros')

x, _ = data.theano_vars()
if debug:
    x.tag.test_value = np.zeros((batch_size, 784), dtype=np.float32)

read = ReadLayer(name='read',
                 parent=['x', 'error'],
                 parent_dim=[784, 784],
                 recurrent=['enc', 'dec'],
                 recurrent_dim=[256, 256],
                 nout=8,
                 glimpse_shape=(batch_size, 1, 2, 2),
                 input_shape=(batch_size, 1, 28, 28),
                 batch_size=batch_size,
                 init_U=InitCell('rand'))
enc = LSTM(name='enc',
           parent=['read'],
           parent_dim=[8],
           recurrent=['dec'],
           recurrent_dim=[256],
           batch_size=batch_size,
           nout=256,
           unit='tanh',
           init_W=init_W,
           init_U=init_U,
           init_b=init_b)
phi_mu = FullyConnectedLayer(name='phi_mu',
                             parent=['enc'],
                             parent_dim=[256],
                             nout=latsz,
                             unit='linear',
                             init_W=init_W,
                             init_b=init_b)
phi_var = RealVectorLayer(name='phi_var',
                          nout=latsz,
                          init_b=init_b)
prior = PriorLayer(name='prior',
                   parent=['phi_mu', 'phi_var'],
                   parent_dim=[latsz, latsz],
                   use_sample=1,
                   nout=latsz)
kl = PriorLayer(name='kl',
                parent=['phi_mu', 'phi_var'],
                   parent_dim=[latsz, latsz],
                use_sample=0,
                tol=1e-4,
                nout=latsz)
dec = LSTM(name='dec',
           parent=['prior'],
           parent_dim=[latsz],
           batch_size=batch_size,
           nout=256,
           unit='tanh',
           init_W=init_W,
           init_U=init_U,
           init_b=init_b)
w1 = FullyConnectedLayer(name='w1',
                         parent=['dec'],
                         parent_dim=[256],
                         nout=4,
                         unit='linear',
                         init_W=init_W,
                         init_b=init_b)
write= WriteLayer(name='write',
                  parent=['w1', 'dec'],
                  parent_dim=[4, 256],
                  nout=784,
                  glimpse_shape=(batch_size, 1, 2, 2),
                  input_shape=(batch_size, 1, 28, 28))
error = ErrorLayer(name='error',
                   parent=['x'],
                   parent_dim=[784],
                   recurrent=['canvas'],
                   recurrent_dim=[784],
                   is_binary=1,
                   nout=inpsz,
                   batch_size=batch_size)
canvas = CanvasLayer(name='canvas',
                     parent=['write'],
                     parent_dim=[784],
                     nout=inpsz,
                     batch_size=batch_size)
nodes = [read, enc, phi_mu, phi_var, prior, kl, dec, w1, write, error, canvas, phi_var]
for node in nodes:
    node.initialize()
params = flatten([node.get_params().values() for node in nodes])
def inner_fn(enc_h_tm1, dec_h_tm1, canvas_h_tm1, x, phi_var_out):
    err_out = error.fprop([[x], [canvas_h_tm1]])
    read_out = read.fprop([[x, err_out], [enc_h_tm1, dec_h_tm1]])
    enc_out = enc.fprop([[read_out], [enc_h_tm1, dec_h_tm1]])
    phi_mu_out = phi_mu.fprop([enc_out])
    prior_out = prior.fprop([phi_mu_out, phi_var_out])
    kl_out = kl.fprop([phi_mu_out, phi_var_out])
    dec_out = dec.fprop([[prior_out], [dec_h_tm1]])
    w1_out = w1.fprop([dec_out])
    write_out = write.fprop([w1_out, dec_out])
    canvas_out = canvas.fprop([[write_out], [canvas_h_tm1]])
    error_out = error.fprop([[x], [canvas_out]])

    return enc_out, dec_out, canvas_out, kl_out

phi_var_out = phi_var.fprop()
((enc_, dec_, canvas_, kl_), updates) = theano.scan(fn=inner_fn,
                                                    outputs_info=[enc.get_init_state(),
                                                                  dec.get_init_state(),
                                                                  canvas.get_init_state(),
                                                                  None],
                                                    non_sequences=[x, phi_var_out],
                                                    n_steps=n_steps)
for k, v in updates.iteritems():
    k.default_update = v
recon_term = NllBin(x, T.nnet.sigmoid(canvas_[-1])).sum()
kl_term = (kl_.sum(axis=2).sum(axis=0)).sum()
cost = recon_term + kl_term
cost.name = 'cost'
recon_term.name = 'recon_term'
kl_term.name = 'kl_term'
recon_err = ((x - T.nnet.sigmoid(canvas_[-1]))**2).mean() / x.std()
recon_err.name = 'recon_err'
model.inputs = [x]
model._params = params
model.nodes =nodes

optimizer = Adam(
    lr=0.001
)

extension = [
    GradientClipping(),
    EpochCount(10000),
    Monitoring(freq=10,
               ddout=[cost, recon_term, kl_term, recon_err],
               data=[Iterator(data, batch_size)]),
    Picklize(freq=1000,
             path=savepath),
    EarlyStopping(freq=200, path=savepath)
]

mainloop = Training(
    name='draw',
    data=Iterator(data, batch_size),
    model=model,
    optimizer=optimizer,
    cost=cost,
    outputs=[cost],
    extension=extension
)
mainloop.run()
