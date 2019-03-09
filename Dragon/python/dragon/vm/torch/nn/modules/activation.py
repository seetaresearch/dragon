# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.nn import Module


class ReLU(Module):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace
        self.register_op()

    def register_op(self):
        self.op_meta = {'op_type': 'Relu', 'arguments':{}}

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str

    def forward(self, x):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [x if self.inplace else self.register_output()]
        return self.run(inputs, outputs)


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01, inplace=False):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'Relu',
            'arguments': {'slope': self.negative_slope}
        }

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'negative_slope={}{}'.format(self.negative_slope, inplace_str)

    def forward(self, x):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [x if self.inplace else self.register_output()]
        return self.run(inputs, outputs)


class ELU(Module):
    def __init__(self, alpha=1.0, inplace=False):
        super(ELU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'Elu',
            'arguments': {'alpha': self.alpha},
        }

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'alpha={}{}'.format(self.alpha, inplace_str)

    def forward(self, x):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [x if self.inplace else self.register_output()]
        return self.run(inputs, outputs)


class SELU(Module):
    def __init__(self, inplace=False):
        super(SELU, self).__init__()
        self.inplace = inplace
        self.register_op()

    def register_op(self):
        self.op_meta = {'op_type': 'SElu', 'arguments': {}}

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str

    def forward(self, x):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [x if self.inplace else self.register_output()]
        return self.run(inputs, outputs)


class Sigmoid(Module):
    def __init__(self, inplace=False):
        super(Sigmoid, self).__init__()
        self.inplace = inplace
        self.register_op()

    def register_op(self):
        self.op_meta = {'op_type': 'Sigmoid', 'arguments': {}}

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str

    def forward(self, x):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [x if self.inplace else self.register_output()]
        return self.run(inputs, outputs)


class Tanh(Module):
    def __init__(self, inplace=False):
        super(Tanh, self).__init__()
        self.inplace = inplace
        self.register_op()

    def register_op(self):
        self.op_meta = {'op_type': 'Tanh', 'arguments': {}}

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str

    def forward(self, x):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [x if self.inplace else self.register_output()]
        return self.run(inputs, outputs)


class Softmax(Module):
    def __init__(self, dim=None, inplace=False):
        super(Softmax, self).__init__()
        self.dim = dim
        self.inplace = inplace
        if dim is None:
            raise ValueError('Excepted a valid dim, got None.')
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'Softmax',
            'arguments': {'axis': self.dim},
        }

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'dim={}{}'.format(self.dim, inplace_str)

    def forward(self, x):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [x if self.inplace else self.register_output()]
        return self.run(inputs, outputs)