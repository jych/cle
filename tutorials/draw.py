import ipdb
import numpy as np
import theano.tensor as T

from cle.cle.data import Iterator
from cle.cle.cost import Gaussian
from cle.cle.graph.net import Net
from cle.cle.models import Model
from cle.cle.models.draw import (
    ReadLayer,
    WriteLayer,
    CanvasLayer
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
from cle.cle.utils import OrderedDict
from cle.datasets.mnist import MNIST


datapath = '/home/junyoung/data/mnist/mnist.pkl'
savepath = '/home/junyoung/repos/cle/saved/'

batch_size = 128
inpsz = 784
latsz = 50
debug = 1

model = Model()
data = MNIST(name='train',
             path=datapath)

init_W = InitCell('rand')
init_U = InitCell('ortho')
init_b = InitCell('zeros')

model.inputs = data.theano_vars()
x, y = model.inputs
if debug:
    x.tag.test_value = np.zeros((batch_size, 784), dtype=np.float32)

inputs = [x]
inputs_dim = {'x':inpsz}
c1 = ConvertLayer(name='c1',
                  parent=['x'],
                  nout=784,
                  outshape=(batch_size, 1, 28, 28))
read = ReadLayer(name='read',
                 parent=['c1', 'error'],
                 recurrent=['dec'],
                 nout=5,
                 N=12,
                 width=28,
                 height=28,
                 batch_size=batch_size,
                 init_U=InitCell('rand'))
c2 = ConvertLayer(name='c2',
                  parent=['read'],
                  nout=784,
                  outshape=(batch_size, 784))
enc = LSTM(name='enc',
           parent=['c2'],
           recurrent=['dec'],
           batch_size=batch_size,
           nout=500,
           unit='tanh',
           init_W=init_W,
           init_U=init_U,
           init_b=init_b)
phi_mu = FullyConnectedLayer(name='phi_mu',
                             parent=['enc'],
                             nout=latsz,
                             unit='linear',
                             init_W=init_W,
                             init_b=init_b)
phi_var = RealVectorLayer(name='phi_var',
                          nout=latsz,
                          init_b=init_b)
prior = PriorLayer(name='prior',
                   parent=['phi_mu', 'phi_var'],
                   use_sample=1,
                   nout=latsz)
kl = PriorLayer(name='kl',
                parent=['phi_mu', 'phi_var'],
                use_sample=0,
                nout=latsz)
dec = LSTM(name='dec',
           parent=['prior'],
           batch_size=batch_size,
           nout=500,
           unit='tanh',
           init_W=init_W,
           init_U=init_U,
           init_b=init_b)
w1 = FullyConnectedLayer(name='w1',
                         parent=['dec'],
                         nout=inpsz,
                         unit='linear',
                         init_W=init_W,
                         init_b=init_b)
c3 = ConvertLayer(name='c3',
                  parent=['w1'],
                  nout=784,
                  outshape=(batch_size, 1, 28, 28))
write= WriteLayer(name='write',
                  parent=['dec'],
                  nout=5,
                  N=12,
                  width=28,
                  height=28,
                  batch_size=batch_size,
                  init_U=init_U)
error = CanvasLayer(name='error',
                    parent=['c1'],
                    recurrent=['canvas'],
                    is_write=0,
                    is_binary=1)
canvas = CanvasLayer(name='canvas',
                     parent=['c3'],
                     is_write=1,
                     is_binary=1,
                     outshape=(batch_size, 1, 28, 28))
nodes = [enc, dec, phi_mu, phi_var, prior, kl, read, write, error, canvas, c1, c2, c3, w1]
draw = Net(inputs=inputs, inputs_dim=inputs_dim, nodes=nodes)
(kl_, canvas_), updates =\
    draw.build_recurrent_graph(output_args=[kl, canvas], nonseq_inputs=[0], n_steps=20)
ipdb.set_trace()

# Dimshuffle to (example, time_step, dimension) or (bs, ts, fd)
ty = y.dimshuffle(1, 0, 2)
dy = T.extra_ops.repeat(ty, num_sample, axis=0)
ry = dy.reshape((dy.shape[0]*dy.shape[1], -1))

tmu = mu.dimshuffle(1, 0, 2)
rmu = tmu.reshape((tmu.shape[0]*tmu.shape[1], -1))    

tkl = kl_.dimshuffle(1, 0, 2)
rkl = tkl.reshape((tkl.shape[0]*tkl.shape[1], -1))
kl_term = rkl.sum(axis=1)[rkl.sum(axis=1).nonzero()]

var = theta_var.fprop()
max_theta_var = T.exp(var).max()
mean_theta_var = T.exp(var).mean()
min_theta_var = T.exp(var).min()
max_theta_var.name = 'max_theta_var'
mean_theta_var.name = 'mean_theta_var'
min_theta_var.name = 'min_theta_var'

max_phi_var = T.exp(phi_var).max()
mean_phi_var = T.exp(phi_var).mean()
min_phi_var = T.exp(phi_var).min()
max_phi_var.name = 'max_phi_var'
mean_phi_var.name = 'mean_phi_var'
min_phi_var.name = 'min_phi_var'

recon = Gaussian(ry, rmu, var, tol=1e-8)
recon = recon.reshape((tmu.shape[0], tmu.shape[1]))
recon = recon.mean(axis=0)
recon_term = recon.mean()
kl_term = kl_term.mean()
recon_err = T.sqrt(T.sqr(ry - rmu).mean()) / ry.std()

cost = recon_term + kl_term
cost.name = 'cost'
recon_term.name = 'recon_term'
kl_term.name = 'kl_term'
recon_err.name = 'recon_err'
vae.add_input([y, mask])
vae.add_node(theta_var)
model.graphs = [vae]

optimizer = Adam(
    lr=0.001
)

extension = [
    GradientClipping(),
    EpochCount(10000),
    Monitoring(freq=50,
               ddout=[cost, recon_term, kl_term, recon_err,
                      max_phi_var, mean_phi_var, min_phi_var,
                      max_theta_var, mean_theta_var, min_theta_var],
               data=[Iterator(data, batch_size_enc)]),
    Picklize(freq=200,
             path=savepath),
    EarlyStopping(path=savepath)
]

mainloop = Training(
    name='draw',
    data=Iterator(data, batch_size_enc),
    model=model,
    optimizer=optimizer,
    cost=cost,
    outputs=[cost],
    extension=extension
)
mainloop.run()
