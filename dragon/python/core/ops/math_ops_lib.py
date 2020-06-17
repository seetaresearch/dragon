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


class Accumulate(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Accumulate, self).__init__(key, dev, **kwargs)
        self.alpha = kwargs.get('alpha', 1.)
        self.beta = kwargs.get('beta', 1.)

    def attributes(self):
        return {
            'op_type': 'Accumulate',
            'arguments': {
                'alpha': self.alpha,
                'beta': self.beta,
            }
        }

    def forward(self, inputs, outputs=None):
        if outputs is None:
            outputs = [self.alloc() for _ in range(len(inputs))]
        return self.dispatch(inputs, outputs, no_grad=True)


class Affine(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Affine, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 1)
        self.num_axes = kwargs.get('num_axes', 1)

    def attributes(self):
        return {
            'op_type': 'Affine',
            'arguments': {
                'axis': self.axis,
                'num_axes': self.num_axes,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class Binary(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Binary, self).__init__(key, dev, **kwargs)
        self.op_type = kwargs.get('op_type', '')

    def attributes(self):
        return {'op_type': self.op_type, 'arguments': {}}

    def forward(self, inputs, outputs=None):
        if outputs is None:
            outputs = [self.alloc()]
        else:
            outputs[0]._device = self.alloc()
        return self.dispatch(inputs, outputs)


class Clip(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Clip, self).__init__(key, dev, **kwargs)
        self.low = kwargs.get('low', None)
        self.high = kwargs.get('high', None)
        if self.low is not None:
            self.low = float(self.low)
        if self.high is not None:
            self.high = float(self.high)

    def attributes(self):
        return {
            'op_type': 'Clip',
            'arguments': {
                'low': self.low,
                'high': self.high,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class Dot(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Dot, self).__init__(key, dev, **kwargs)
        self.transA = kwargs.get('transA', False)
        self.transB = kwargs.get('transB', False)

    def attributes(self):
        return {
            'op_type': 'Dot',
            'arguments': {
                'transA': self.transA,
                'transB': self.transB,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class FullyConnected(Operator):
    def __init__(self, key, dev, **kwargs):
        super(FullyConnected, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 1)
        self.transW = kwargs.get('transW', True)

    def attributes(self):
        return {
            'op_type': 'FullyConnected',
            'arguments': {
                'axis': self.axis,
                'transW': self.transW,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class Matmul(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Matmul, self).__init__(key, dev, **kwargs)
        self.transA = kwargs.get('transA', False)
        self.transB = kwargs.get('transB', False)

    def attributes(self):
        return {
            'op_type': 'Matmul',
            'arguments': {
                'transA': self.transA,
                'transB': self.transB,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class Unary(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Unary, self).__init__(key, dev, **kwargs)
        self.op_type = kwargs.get('op_type', '')

    def attributes(self):
        return {'op_type': self.op_type, 'arguments': {}}

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])
