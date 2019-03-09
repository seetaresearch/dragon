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

from dragon.vm.torch.ops.modules.base import BaseModule


class _InitModule(BaseModule):
    def __init__(self, key, ctx, **kwargs):
        super(_InitModule, self).__init__(key, ctx, **kwargs)
        self.n_dim = kwargs.get('n_dim', 0)
        self.dtype = kwargs.get('dtype', 'float32')

    def update_arguments(self, A, shape):
        for i, e in enumerate(shape):
            self.set_argument_i64('{}/dims[{}]'.format(A, i), e)

    def forward(self, x, shape):
        outputs = [x]; self.unify_devices(outputs)
        callback = lambda A: self.update_arguments(A, shape)
        return self.run([], outputs, callback=callback)


class Fill(_InitModule):
    def __init__(self, key, ctx, **kwargs):
        super(Fill, self).__init__(key, ctx, **kwargs)
        self.value = kwargs.get('value', 0.0)
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'Fill',
            'arguments': {
                'dtype': self.dtype,
                'value': float(self.value),
                'dims_desc': [
                    '${{ANCHOR}}/dims[{}]'.format(n)
                        for n in range(self.n_dim)
                ],
            },
        }


class RandomNormal(_InitModule):
    def __init__(self, key, ctx, **kwargs):
        super(RandomNormal, self).__init__(key, ctx, **kwargs)
        self.mean = kwargs.get('mean', 0.0)
        self.std = kwargs.get('std', 1.0)
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'RandomNormal',
            'arguments': {
                'dtype': self.dtype,
                'mean': float(self.mean),
                'std': float(self.std),
                'dims_desc': [
                    '${{ANCHOR}}/dims[{}]'.format(n)
                        for n in range(self.n_dim)
                ],
            },
        }


class RandomUniform(_InitModule):
    def __init__(self, key, ctx, **kwargs):
        super(RandomUniform, self).__init__(key, ctx, **kwargs)
        self.low = kwargs.get('low', 0.0)
        self.high = kwargs.get('high', 1.0)
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'RandomUniform',
            'arguments': {
                'dtype': self.dtype,
                'low': float(self.low),
                'high': float(self.high),
                'dims_desc': [
                    '${{ANCHOR}}/dims[{}]'.format(n)
                        for n in range(self.n_dim)
                ],
            },
        }