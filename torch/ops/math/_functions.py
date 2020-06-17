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

from dragon.vm.torch.autograd import function


class Accumulate(function.Function):
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
            },
        }

    def forward(self, input, out=None):
        out = out if out else self.alloc()
        return self.dispatch([input], [out], no_grad=True)


class Binary(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(Binary, self).__init__(key, dev, **kwargs)
        self.op_type = kwargs.get('op_type', '')

    def attributes(self):
        return {'op_type': self.op_type, 'arguments': {}}

    def forward(self, input, value, out=None):
        out = out if out else self.alloc()
        return self.dispatch([input, value], [out])


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
        out = out if out else self.alloc()
        return self.dispatch([input], [out])


class Unary(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(Unary, self).__init__(key, dev, **kwargs)
        self.op_type = kwargs.get('op_type', '')

    def attributes(self):
        return {'op_type': self.op_type, 'arguments': {}}

    def forward(self, input, out=None):
        out = out if out else self.alloc()
        return self.dispatch([input], [out])


class MM(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(MM, self).__init__(key, dev, **kwargs)
        self.transA = kwargs.get('transA', False)
        self.transB = kwargs.get('transB', False)

    def attributes(self):
        return {
            'op_type': 'Matmul',
            'arguments': {
                'transA': self.transA,
                'transB': self.transB,
            },
        }

    def forward(self, mat1, mat2, out=None):
        out = out if out else self.alloc()
        return self.dispatch([mat1, mat2], [out])
