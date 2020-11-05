# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Activation ops library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework.ops import Operator


class Activation(Operator):
    """Base activation operator."""

    def __init__(self, key, dev, **kwargs):
        super(Activation, self).__init__(key, dev, **kwargs)
        self.op_type = kwargs.get('op_type', '')

    def attributes(self):
        return {'op_type': self.op_type, 'arguments': {}}

    def forward(self, inputs, inplace=False):
        outputs = [self.alloc(inputs[0]) if inplace else self.alloc()]
        return self.dispatch(inputs, outputs)


class Dropout(Operator):
    """Dropout operator."""

    def __init__(self, key, dev, **kwargs):
        super(Dropout, self).__init__(key, dev, **kwargs)

    def attributes(self):
        return {
            'op_type': 'Dropout',
            'arguments': {'ratio_desc': '${HANDLE}/ratio'},
        }

    def feed(self, ws, handle, ratio):
        self.feed_arg(ws, '{}/ratio'.format(handle), ratio, 'float32')

    def forward(self, inputs, ratio, inplace=False):
        outputs = [self.alloc(inputs[0]) if inplace else self.alloc()]
        return self.dispatch(inputs, outputs,
                             callback=lambda ws, handle:
                             self.feed(ws, handle, ratio))


class DropBlock2d(Dropout):
    """DropBlock2d operator."""

    def __init__(self, key, dev, **kwargs):
        super(DropBlock2d, self).__init__(key, dev, **kwargs)
        self.block_size = kwargs.get('block_size', 7)
        self.data_format = kwargs.get('data_format', 'NCHW')

    def attributes(self):
        return {
            'op_type': 'DropBlock2d',
            'arguments': {
                'ratio_desc': '${HANDLE}/ratio',
                'block_size': self.block_size,
                'data_format': self.data_format,
            },
        }


class DropPath(Dropout):
    """DropPath operator."""

    def __init__(self, key, dev, **kwargs):
        super(DropPath, self).__init__(key, dev, **kwargs)

    def attributes(self):
        return {
            'op_type': 'DropPath',
            'arguments': {'ratio_desc': '${HANDLE}/ratio'},
        }


class Elu(Activation):
    """Elu operator."""

    def __init__(self, key, dev, **kwargs):
        super(Elu, self).__init__(key, dev, **kwargs)
        self.alpha = kwargs.get('alpha', 1.)

    def attributes(self):
        return {
            'op_type': 'Elu',
            'arguments': {'alpha': float(self.alpha)},
        }


class HardSigmoid(Activation):
    """HardSigmoid operator."""

    def __init__(self, key, dev, **kwargs):
        super(HardSigmoid, self).__init__(key, dev, **kwargs)
        self.alpha = kwargs.get('alpha', 0.2)
        self.beta = kwargs.get('beta', 0.5)

    def attributes(self):
        return {
            'op_type': 'HardSigmoid',
            'arguments': {
                'alpha': float(self.alpha),
                'beta': float(self.beta),
            },
        }


class HardSwish(Activation):
    """HardSwish operator."""

    def __init__(self, key, dev, **kwargs):
        super(HardSwish, self).__init__(key, dev, **kwargs)
        self.alpha = kwargs.get('alpha', 0.2)
        self.beta = kwargs.get('beta', 0.5)

    def attributes(self):
        return {
            'op_type': 'HardSwish',
            'arguments': {
                'alpha': float(self.alpha),
                'beta': float(self.beta),
            },
        }


class PRelu(Operator):
    """PRelu operator."""

    def __init__(self, key, dev, **kwargs):
        super(PRelu, self).__init__(key, dev, **kwargs)
        self.data_format = kwargs.get('data_format', 'NCHW')

    def attributes(self):
        return {
            'op_type': 'PRelu',
            'arguments': {'data_format': self.data_format},
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class Relu(Activation):
    """Relu operator."""

    def __init__(self, key, dev, **kwargs):
        super(Relu, self).__init__(key, dev, **kwargs)
        self.alpha = kwargs.get('alpha', 0.)

    def attributes(self):
        return {
            'op_type': 'Relu',
            'arguments': {'alpha': float(self.alpha)},
        }


class Relu6(Activation):
    """Relu6 operator."""

    def __init__(self, key, dev, **kwargs):
        super(Relu6, self).__init__(key, dev, **kwargs)

    def attributes(self):
        return {
            'op_type': 'Relu',
            'arguments': {'max_value': 6.},
        }


class Selu(Activation):
    """Selu operator."""

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
    """Softmax operator."""

    def __init__(self, key, dev, **kwargs):
        super(Softmax, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 1)

    def attributes(self):
        return {
            'op_type': 'Softmax',
            'arguments': {'axis': self.axis},
        }
