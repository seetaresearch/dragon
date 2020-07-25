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
"""Math functions library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.core.autograd import function


class Axpby(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(Axpby, self).__init__(key, dev, **kwargs)
        self.alpha = kwargs.get('alpha', 1.)
        self.beta = kwargs.get('beta', 1.)

    def attributes(self):
        return {
            'op_type': 'Axpby',
            'arguments': {
                'alpha': self.alpha,
                'beta': self.beta,
            },
        }

    def forward(self, input, out=None):
        return self.dispatch([input], [self.alloc(out)], no_grad=True)


class BinaryFunc(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(BinaryFunc, self).__init__(key, dev, **kwargs)
        self.op_type = kwargs.get('op_type', '')

    def attributes(self):
        return {'op_type': self.op_type, 'arguments': {}}

    def forward(self, input, value, out=None):
        return self.dispatch([input, value], [self.alloc(out)])


class Clip(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(Clip, self).__init__(key, dev, **kwargs)
        self.min = kwargs.get('min', None)
        self.max = kwargs.get('max', None)
        if self.min is not None:
            self.min = float(self.min)
        if self.max is not None:
            self.max = float(self.max)

    def attributes(self):
        return {
            'op_type': 'Clip',
            'arguments': {
                'low': self.min,
                'high': self.max,
            },
        }

    def forward(self, input, out=None):
        return self.dispatch([input], [self.alloc(out)])


class UnaryFunc(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(UnaryFunc, self).__init__(key, dev, **kwargs)
        self.op_type = kwargs.get('op_type', '')

    def attributes(self):
        return {'op_type': self.op_type, 'arguments': {}}

    def forward(self, input, out=None):
        return self.dispatch([input], [self.alloc(out)])


class MatMul(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(MatMul, self).__init__(key, dev, **kwargs)
        self.transpose_a = kwargs.get('transpose_a', False)
        self.transpose_b = kwargs.get('transpose_b', False)

    def attributes(self):
        return {
            'op_type': 'MatMul',
            'arguments': {
                'transA': self.transpose_a,
                'transB': self.transpose_b,
            },
        }

    def forward(self, mat1, mat2, out=None):
        return self.dispatch([mat1, mat2], [self.alloc(out)])
