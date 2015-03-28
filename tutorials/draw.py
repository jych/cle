import ipdb
import numpy as np
import theano
import theano.tensor as T

from cle.cle.data import Iterator
from cle.cle.cost import NllBin
from cle.cle.models import Model
from cle.cle.models.draw import (
    CanvasLayer,
    ErrorLayer,
    ReadLayer,
    WriteLayer
)
from cle.cle.layers import InitCell
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


datapath = '/home/junyoung/data/mnist/mnist_binarized_salakhutdinov.pkl'
savepath = '/home/junyoung/repos/cle/saved/'

batch_size = 100
inpsz = 784
latsz = 100
n_steps = 64
debug = 0

model = Model()
data = MNIST(name='train',
             unsupervised=1,
             path=datapath)

init_W = InitCell('rand')
init_U = InitCell('ortho')
init_b = InitCell('zeros')
init_b_sig = InitCell('const', mean=0.)

x, _ = data.theano_vars()
if debug:
    x.tag.test_value = np.zeros((batch_size, 784), dtype=np.float32)

read_param = FullyConnectedLayer(name='read_param',
                                 parent=['dec_tm1'],
                                 parent_dim=[256],
                                 nout=4,
                                 unit='linear',
                                 init_W=init_W,
                                 init_b=init_b)
read_param_sig = FullyConnectedLayer(name='read_sig',
                                     parent=['dec_tm1'],
                                     parent_dim=[256],
                                     nout=1,
                                     unit='softplus',
                                     init_W=init_W,
                                     init_b=init_b_sig)
read = ReadLayer(name='read',
                 parent=['x', 'error', 'read_param', 'read_param_sig'],
                 glimpse_shape=(batch_size, 1, 2, 2),
                 input_shape=(batch_size, 1, 28, 28))
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
phi_sig = FullyConnectedLayer(name='phi_sig',
                              parent=['enc'],
                              parent_dim=[256],
                              nout=latsz,
                              unit='softplus',
                              init_W=init_W,
                              init_b=init_b_sig)
prior = PriorLayer(name='prior',
                   parent=['phi_mu', 'phi_sig'],
                   parent_dim=[latsz, latsz],
                   use_sample=1,
                   nout=latsz)
kl = PriorLayer(name='kl',
                parent=['phi_mu', 'phi_sig'],
                parent_dim=[latsz, latsz],
                use_sample=0,
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
w = FullyConnectedLayer(name='w',
                        parent=['dec'],
                        parent_dim=[256],
                        nout=25,
                        unit='linear',
                        init_W=init_W,
                        init_b=init_b)
write_param = FullyConnectedLayer(name='write_param',
                                  parent=['dec'],
                                  parent_dim=[256],
                                  nout=4,
                                  unit='linear',
                                  init_W=init_W,
                                  init_b=init_b)
write_param_sig = FullyConnectedLayer(name='write_sig',
                                      parent=['dec_tm1'],
                                      parent_dim=[256],
                                      nout=1,
                                      unit='softplus',
                                      init_W=init_W,
                                      init_b=init_b_sig)
write = WriteLayer(name='write',
                   parent=['w', 'write_param', 'write_param_sig'],
                   glimpse_shape=(batch_size, 1, 5, 5),
                   input_shape=(batch_size, 1, 28, 28))
error = ErrorLayer(name='error',
                   parent=['x'],
                   recurrent=['canvas'],
                   is_binary=1,
                   batch_size=batch_size)
canvas = CanvasLayer(name='canvas',
                     parent=['write'],
                     nout=784,
                     batch_size=batch_size)
nodes = [read_param, read_param_sig, read, enc, phi_mu, phi_sig, prior, kl, dec, w, write_param, write_param_sig, write, error, canvas]
for node in nodes:
    node.initialize()
params = flatten([node.get_params().values() for node in nodes])

def inner_fn(enc_tm1, dec_tm1, canvas_tm1, x):

    err_out = error.fprop([[x], [canvas_tm1]])
    read_param_out = read_param.fprop([dec_tm1])
    read_param_sig_out = read_param_sig.fprop([dec_tm1])
    read_out = read.fprop([x, err_out, read_param_out, read_param_sig_out])
    enc_out = enc.fprop([[read_out], [enc_tm1, dec_tm1]])
    phi_mu_out = phi_mu.fprop([enc_out])
    phi_sig_out = phi_sig.fprop([enc_out])
    prior_out = prior.fprop([phi_mu_out, phi_sig_out])
    kl_out = kl.fprop([phi_mu_out, phi_sig_out])
    dec_out = dec.fprop([[prior_out], [dec_tm1]])
    w_out = w.fprop([dec_out])
    write_param_out = write_param.fprop([dec_out])
    write_param_sig_out = write_param_sig.fprop([dec_out])
    write_out = write.fprop([w_out, write_param_out, write_param_sig_out])
    canvas_out = canvas.fprop([[write_out], [canvas_tm1]])

    return enc_out, dec_out, canvas_out, kl_out

((enc_out, dec_out, canvas_out, kl_out), updates) =\
    theano.scan(fn=inner_fn,
                outputs_info=[enc.get_init_state(),
                              dec.get_init_state(),
                              canvas.get_init_state(),
                              None],
                non_sequences=[x],
                n_steps=n_steps)
for k, v in updates.iteritems():
    k.default_update = v

recon_term = NllBin(x, T.nnet.sigmoid(canvas_out[-1])).mean()
kl_term = kl_out.sum(axis=0).mean()
cost = recon_term + kl_term
cost.name = 'cost'
recon_term.name = 'recon_term'
kl_term.name = 'kl_term'
recon_err = ((x - T.nnet.sigmoid(canvas_out[-1]))**2).mean() / x.std()
recon_err.name = 'recon_err'
model.inputs = [x]
model._params = params
model.nodes = nodes

optimizer = Adam(
    lr=0.001
)

extension = [
    GradientClipping(batch_size=batch_size),
    EpochCount(10000),
    Monitoring(freq=100,
               ddout=[cost, recon_term, kl_term, recon_err],
               data=[Iterator(data, batch_size)]),
    Picklize(freq=2000,
             path=savepath),
    EarlyStopping(freq=500, path=savepath)
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
