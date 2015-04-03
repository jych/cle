import ipdb
import theano.tensor as T

from cle.cle.layers import StemCell
from cle.cle.utils import tolist, totuple, unpack
from theano.tensor.nnet import conv2d


class Conv2DLayer(StemCell):
    """
    2D convolutional layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 parshape=[],
                 outshape=None,
                 filtershape=None,
                 tied_bias=True,
                 step_size=(1, 1),
                 border_mode='valid',
                 **kwargs):
        super(Conv2DLayer, self).__init__(**kwargs)
        # Shape should be (batch_size, num_channels, x, y)
        if (outshape is None and filtershape is None) or\
                (outshape is not None and filtershape is not None):
            raise ValueError("Either outshape or filtershape should be given,\
                             but don't provide both of them.")
        self.outshape = outshape
        self.filtershape = filtershape
        self.tied_bias = tied_bias
        self.step_size = totuple(step_size)
        self.border_mode = border_mode
        for i, par in enumerate(tolist(self.parent.keys())):
             if len(parshape) != 0:
                self.parent[par] = parshape[i]
           
    def fprop(self, x):
        # Conv layer can have only one parent.
        # Later, we will extend to generalize this.
        # For now, we satisfy with fullyconnected layer
        # that can embed multiple conv layer parents
        # into same hidden space.
        x = unpack(x)
        parname, parshape = unpack(self.parent.items())
        z = T.zeros(self.outshape)
        W = self.params['W_'+parname+'__'+self.name]
        z += conv2d(
            x, W,
            image_shape=parshape,
            subsample=self.step_size,
            border_mode=self.border_mode,
            filter_shape=self.filtershape
        )
        if self.tied_bias:
            z += self.params['b_'+self.name].dimshuffle('x', 0, 'x', 'x')
        else:
            z += self.params['b_'+self.name].dimshuffle('x', 0, 1, 2)
        z = self.nonlin(z)
        z.name = self.name
        return z

    def initialize(self):
        parname, parshape = unpack(self.parent.items())
        outshape = self.outshape
        filtershape = self.filtershape
        batch_size = parshape[0]
        nchannels = parshape[1]
        if filtershape is not None:
            nfilters = filtershape[1]
            if self.border_mode == 'valid':
                x = parshape[2] - filtershape[2] + 1
                y = parshape[3] - filtershape[3] + 1
            else:
                x = parshape[2] + filtershape[2] - 1
                y = parshape[3] + filtershape[3] - 1
            self.outshape = (batch_size, nfilters, x, y)
        else:
            nfilters = outshape[1]
            if self.border_mode == 'valid':
                x = parshape[2] - outshape[2] + 1
                y = parshape[3] - outshape[3] + 1
            elif self.border_mode == 'full':
                x = outshape[2] - parshape[2] + 1
                y = outshape[3] - parshape[3] + 1
            W_shape = (nfilters, nchannels, x, y)
            self.filtershape = W_shape
        W_name = 'W_'+parname+'__'+self.name
        self.alloc(self.init_W.get(self.filtershape, W_name))
        b_name = 'b_'+self.name
        if self.tied_bias:
            b_shape = nfilters
            self.alloc(self.init_b.get(b_shape, b_name))
        else:
            b_shape = (nfilters, x, y)
            self.alloc(self.init_b.get(b_shape, b_name))


class ConvertLayer(StemCell):
    """
    Convert 2D matrix to 4D tensor
    or vice versa

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 outshape=None,
                 axes=('b', 'c', 'x', 'y'),
                 **kwargs):
        super(ConvertLayer, self).__init__(**kwargs)
        self.outshape = outshape
        if len(outshape) == 2 or outshape is None:
            convert_type = 'convert2matrix'
            self.nout = outshape[1]
        elif len(outshape) == 4:
            convert_type = 'convert2tensor4'
        self.fprop = self.which_fn(convert_type)
        self.convert_type = convert_type
        self.axes = axes

    def convert2matrix(self, x):
        x = unpack(x)
        # Assume that axes of x is always ('b', 'c', 'x', 'y')
        refaxes = ('b', 'c', 'x', 'y')
        newaxes = ()
        for axis in self.axes:
            newaxes += (refaxes.index(axis),)
        x = x.dimshuffle(newaxes)
        z = x.reshape((x.shape[0], T.prod(x.shape[1:])))
        z.name = self.name
        return z

    def convert2tensor4(self, x):
        x = unpack(x)
        z = x.reshape(self.outshape)
        z.name = self.name
        return z

    def initialize(self):
        pass

    def __getstate__(self):
        dic = self.__dict__.copy()
        dic.pop('fprop')
        return dic

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.fprop = self.which_convert(self.convert_type)
