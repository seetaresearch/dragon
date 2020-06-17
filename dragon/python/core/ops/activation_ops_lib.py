# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework.ops import Operator


class Activation(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Activation, self).__init__(key, dev, **kwargs)
        self.op_type = kwargs.get('op_type', '')

    def attributes(self):
        return {'op_type': self.op_type, 'arguments': {}}

    def forward(self, inputs, inplace=False):
        outputs = [inputs[0] if inplace else self.alloc()]
        return self.dispatch(inputs, outputs)


class Dropout(Activation):
    def __init__(self, key, dev, **kwargs):
        super(Dropout, self).__init__(key, dev, **kwargs)
        self.prob = kwargs.get('prob', 0.5)
        self.scale = kwargs.get('scale', True)

    def attributes(self):
        return {
            'op_type': 'Dropout',
            'arguments': {
                'prob': self.prob,
                'scale': self.scale,
            }
        }


class DropBlock2d(Operator):
    def __init__(self, key, dev, **kwargs):
        super(DropBlock2d, self).__init__(key, dev, **kwargs)
        self.block_size = kwargs.get('block_size', 7)
        self.keep_prob = kwargs.get('keep_prob', 0.9)
        self.alpha = kwargs.get('alpha', 1.)
        self.decrement = kwargs.get('decrement', 0.)
        self.data_format = kwargs.get('data_format', 'NCHW')

    def attributes(self):
        return {
            'op_type': 'DropBlock2d',
            'arguments': {
                'block_size': self.block_size,
                'keep_prob': self.keep_prob,
                'alpha': self.alpha,
                'decrement': self.decrement,
                'data_format': self.data_format,
            },
        }

    def forward(self, inputs, inplace=False):
        outputs = [inputs[0] if inplace else self.alloc()]
        return self.dispatch(inputs, outputs)


class DropPath(Activation):
    def __init__(self, key, dev, **kwargs):
        super(DropPath, self).__init__(key, dev, **kwargs)
        self.prob = kwargs.get('prob', 0.2)
        self.increment = kwargs.get('increment', 0.)

    def attributes(self):
        return {
            'op_type': 'DropPath',
            'arguments': {
                'prob': self.prob,
                'increment': self.increment,
            }
        }


class Elu(Activation):
    def __init__(self, key, dev, **kwargs):
        super(Elu, self).__init__(key, dev, **kwargs)
        self.alpha = kwargs.get('alpha', 1.)

    def attributes(self):
        return {
            'op_type': 'Elu',
            'arguments': {
                'alpha': float(self.alpha),
            }
        }


class PRelu(Operator):
    def __init__(self, key, dev, **kwargs):
        super(PRelu, self).__init__(key, dev, **kwargs)
        self.data_format = kwargs.get('data_format', 'NCHW')

    def attributes(self):
        return {
            'op_type': 'PRelu',
            'arguments': {
                'data_format': self.data_format,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class Relu(Activation):
    def __init__(self, key, dev, **kwargs):
        super(Relu, self).__init__(key, dev, **kwargs)
        self.alpha = kwargs.get('alpha', 0.)

    def attributes(self):
        return {
            'op_type': 'Relu',
            'arguments': {
                'alpha': float(self.alpha),
            }
        }


class Relu6(Activation):
    def __init__(self, key, dev, **kwargs):
        super(Relu6, self).__init__(key, dev, **kwargs)

    def attributes(self):
        return {
            'op_type': 'Relu',
            'arguments': {
                'max_value': 6.,
            }
        }


class Selu(Activation):
    def __init__(self, key, dev, **kwargs):
        super(Selu, self).__init__(key, dev, **kwargs)
        self.alpha = kwargs.get('alpha', 1.67326)
        self.gamma = kwargs.get('gamma', 1.0507)

    def attributes(self):
        return {
            'op_type': 'Selu',
            'arguments': {
                'alpha': float(self.alpha),
                'gamma': float(self.gamma),
            }
        }


class Softmax(Activation):
    def __init__(self, key, dev, **kwargs):
        super(Softmax, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 1)

    def attributes(self):
        return {
            'op_type': 'Softmax',
            'arguments': {
                'axis': self.axis,
            }
        }
