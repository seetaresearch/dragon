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
"""Normalization ops library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework.ops import Operator


class BatchNorm(Operator):
    def __init__(self, key, dev, **kwargs):
        super(BatchNorm, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', -1)
        self.momentum = kwargs.get('momentum', 0.9)
        self.epsilon = kwargs.get('epsilon', 1e-5)
        self.use_stats = kwargs.get('use_stats', 0)
        if self.use_stats not in (0, 1):
            raise ValueError('Excepted determined stats mode.')

    def attributes(self):
        return {
            'op_type': 'BatchNorm',
            'arguments': {
                'axis': self.axis,
                'momentum': self.momentum,
                'epsilon': self.epsilon,
                'use_stats': self.use_stats,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class GroupNorm(Operator):
    def __init__(self, key, dev, **kwargs):
        super(GroupNorm, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', -1)
        self.group = kwargs.get('group', 32)
        self.epsilon = kwargs.get('epsilon', 1e-5)

    def attributes(self):
        return {
            'op_type': 'GroupNorm',
            'arguments': {
                'axis': self.axis,
                'group': self.group,
                'epsilon': self.epsilon,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class LpNormalize(Operator):
    def __init__(self, key, dev, **kwargs):
        super(LpNormalize, self).__init__(key, dev, **kwargs)
        self.p = kwargs.get('p', 2)
        self.axis = kwargs.get('axis', 0)
        self.num_axes = kwargs.get('num_axes', -1)
        self.epsilon = kwargs.get('epsilon', 1e-12)
        self.reduction = kwargs.get('reduction', 'SUM')

    def attributes(self):
        return {
            'op_type': 'LpNormalize',
            'arguments': {
                'p': self.p,
                'axis': self.axis,
                'num_axes': self.num_axes,
                'epsilon': self.epsilon,
                'reduction': self.reduction,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class LocalResponseNorm(Operator):
    def __init__(self, key, dev, **kwargs):
        super(LocalResponseNorm, self).__init__(key, dev, **kwargs)
        self.size = kwargs.get('size', 5)
        self.alpha = kwargs.get('alpha', 0.0001)
        self.beta = kwargs.get('beta', 0.75)
        self.bias = kwargs.get('bias', 1.)
        self.data_format = kwargs.get('data_format', 'NCHW')

    def attributes(self):
        return {
            'op_type': 'LRN',
            'arguments': {
                'size': self.size,
                'alpha': self.alpha,
                'beta': self.beta,
                'bias': self.bias,
                'data_format': self.data_format,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class SyncBatchNorm(BatchNorm):
    def __init__(self, key, dev, **kwargs):
        super(SyncBatchNorm, self).__init__(key, dev, **kwargs)
        self.process_group = kwargs.get('process_group', None)

    def attributes(self):
        attrs = BatchNorm.attributes(self)
        attrs['op_type'] = 'SyncBatchNorm'
        attrs['arguments'].update(self.process_group.arguments)
        return attrs
