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
    """Axpby function."""

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
    """Binary function."""

    def __init__(self, key, dev, **kwargs):
        super(BinaryFunc, self).__init__(key, dev, **kwargs)
        self.op_type = kwargs.get('op_type', '')

    def attributes(self):
        return {'op_type': self.op_type, 'arguments': {}}

    def forward(self, input, value, out=None):
        return self.dispatch([input, value], [self.alloc(out)])


class Clip(function.Function):
    """Clip function."""

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


class Gemm(function.Function):
    """Gemm function."""

    def __init__(self, key, dev, **kwargs):
        super(Gemm, self).__init__(key, dev, **kwargs)
        self.alpha = kwargs.get('alpha', 1.0)
        self.beta = kwargs.get('beta', 1.0)
        self.transA = kwargs.get('transA', False)
        self.transB = kwargs.get('transB', False)

    def attributes(self):
        return {
            'op_type': 'Gemm',
            'arguments': {
                'axis': -1,
                'alpha': self.alpha,
                'beta': self.beta,
                'transA': self.transA,
                'transB': self.transB,
            },
        }

    def forward(self, mat1, mat2, mat3=None, out=None):
        inputs = [mat1, mat2] + ([mat3] if mat3 else [])
        return self.dispatch(inputs, [self.alloc(out)])


class UnaryFunc(function.Function):
    """Unary function."""

    def __init__(self, key, dev, **kwargs):
        super(UnaryFunc, self).__init__(key, dev, **kwargs)
        self.op_type = kwargs.get('op_type', '')

    def attributes(self):
        return {'op_type': self.op_type, 'arguments': {}}

    def forward(self, input, out=None):
        return self.dispatch([input], [self.alloc(out)])
