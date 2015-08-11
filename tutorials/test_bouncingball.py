import ipdb
import numpy as np
import theano
import theano.tensor as T

from cle.cle.utils import unpickle, tolist, OrderedDict
from cle.datasets.bouncing_balls import BouncingBalls


#data_path = '$your_data_path'
data_path = '/data/lisatmp3/chungjun/bouncing_balls/bouncing_ball_2balls_16wh_20len_50000cases.npy'
#save_path = '$your_model_path'
save_path = '/u/chungjun/src/cle/saved/'
#pkl_name = '$your_model_name'
pkl_name = 'toy_bb_lstm.pkl'

frame_size = 256
# How many examples you want to proceed at a time
batch_size = 100
debug = 0

test_data = BouncingBalls(name='test',
                          path=data_path)

x = test_data.theano_vars()
if debug:
    x.tag.test_value = np.zeros((15, batch_size, frame_size), dtype=np.float32)

exp = unpickle(save_path + pkl_name)
nodes = exp.model.nodes
names = [node.name for node in nodes]

[h1, h2, h3, h4] = nodes

s1_0 = h1.get_init_state(batch_size)
s2_0 = h2.get_init_state(batch_size)
s3_0 = h3.get_init_state(batch_size)

x = T.fmatrix()
ts = T.iscalar()

def inner_fn(s1_tm1, s2_tm1, s3_tm1, iter_):

    s1_t = h1.fprop([[iter_], [s1_tm1]])
    s2_t = h2.fprop([[s1_t], [s2_tm1]])
    s3_t = h3.fprop([[s2_t], [s3_tm1]])
    y_hat_t = h4.fprop([s1_t, s2_t, s3_t])

    return s1_t, s2_t, s3_t, y_hat_t

((s1_temp, s2_temp, s3_temp, y_hat_temp), updates) =\
    theano.scan(fn=inner_fn,
                outputs_info=[s1_0, s2_0, s3_0, x],
                n_steps=ts)

for k, v in updates.iteritems():
    k.default_update = v

test_fn = theano.function(inputs=[x, ts],
                          outputs=[y_hat_temp],
                          updates=updates,
                          allow_input_downcast=True,
                          on_unused_input='ignore')

seed = np.zeros((batch_size, frame_size), dtype=np.float32)
dummy_ts = 200
samples = test_fn(seed, dummy_ts)[-1]
samples = np.transpose(samples, (1, 0, 2))
ipdb.set_trace()
samples = samples.reshape(batch_size, -1)
